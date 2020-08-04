from collections import OrderedDict

import pandas as pd
import numpy as np
from pyterrier.transformer import EstimatorBase
from gensim.models.keyedvectors import KeyedVectors



import random
import os
import pickle
import tensorflow as tf


from preprocess.matrix import similarity_matrix, kernel_from_matrix, hist_from_matrix, kernal_mus, kernel_sigmas
from utils.file_operation import write_result_to_trec_format, parse_stoplist, make_directory
from model.nprf_drmm_config import NPRFDRMMConfig
from model.nprf_drmm import NPRFDRMM
from model.model import NBatchLogger
from metrics.rank_losses import rank_hinge_loss
from metrics.evaluations import evaluate_trec
from utils.nprf_drmm_pair_generator import NPRFDRMMPairGenerator
from utils.result import Result
from utils.relevance_info import Relevance
from tqdm import tqdm

class NeuralPRFEstimator(EstimatorBase):

    def __init__(self, index, data_dir='data/'):
        
        super().__init__()
        self.index = index
        self.nb_docs = self.index.getCollectionStatistics().getNumberOfDocuments()
        self.tempdir = data_dir 
        self.relevance_file = os.path.join(self.tempdir, "relevance.pkl")
        self.topk_idf_file = os.path.join(self.tempdir, "topk.idf.pkl" )
        self.output_file = os.path.join(self.tempdir, "output.txt" )
        self.model_file = os.path.join(self.tempdir, "model.h5" )
        self.hist_path = os.path.join(self.tempdir, 'hist/')
        self.sim_output_path = os.path.join(self.tempdir, 'sim/')
        self.kernel_output_path = os.path.join(self.tempdir, 'ker/')
        
        
    def _get_contents(self, docid):
    # you have code for that
        content=[]
        di = self.index.getDirectIndex()
        doi = self.index.getDocumentIndex()
        lex = self.index.getLexicon()
        postings = di.getPostings(doi.getDocumentEntry(docid))
        for p in postings:
            lee = lex.getLexiconEntry(p.getId())
            words = [lee.getKey()] * p.getFrequency()
            content.extend(words)
        return content
    def create_relevance(self, res, qrels):
        relevance_dict = OrderedDict()
        s1 = pd.merge(res, qrels, how='left', on=['qid','docno'])
        s1.label = s1.label.fillna(0)
        for qid, n in s1.groupby('qid'):
            
            all_res = n.docid.tolist()
            all_score = n.score.tolist()
            # supervised_docid_list = n.docid.tolist()
            # supervised_score_list = n.score.tolist()
            list_rev_0 = n[(n.label==0) ]['docid'].tolist()
            list_rev_1 = n[(n.label>0) & (n.label <2)]['docid'].tolist()
            list_rev_2 = n[n.label>=2]['docid'].tolist()
            judged_docid_list_within_supervised = [list_rev_0, list_rev_1, list_rev_2]
            relevance = Relevance(qid, judged_docid_list_within_supervised, all_res, all_score)
            relevance_dict.update({qid: relevance})
        self.relevance_dict = relevance_dict
        pickle.dump(self.relevance_dict, open(self.relevance_file,'wb'))
        return relevance_dict

    def top_k(self, res, topk=5 ):
        doi = self.index.getDocumentIndex()
        # doc_len_list=[]
        # topk_list=[]
        # docid_list=[]
        doc_top_k = OrderedDict()
        for docid in res.docid.unique().tolist():
            topk_terms=self.top_k_doc(docid,topk)
            doclen=doi.getDocumentLength(docid)
            doc_top_k[docid] = [doclen, topk_terms]
        
        return doc_top_k

    def top_k_doc(self, docid, topk=5):
        di = self.index.getDirectIndex()
        doi = self.index.getDocumentIndex()
        lex = self.index.getLexicon()
        postings = di.getPostings(doi.getDocumentEntry(docid))
        term_list = []
        score_list = []
        for p in postings:
            lee = lex.getLexiconEntry(p.getId())
            # key = lee.getKey() #self.index.getLexicon().getLexiconEntry(p.getId()).key
            count = p.getFrequency()
            dfreq = lee.getValue().getDocumentFrequency()#self.index.getLexicon()[key].getDocumentFrequency()
            idf = np.log((self.nb_docs - dfreq + 0.5) / (dfreq + 0.5))
            tfidf = count * idf
            term_list.append(lee.getKey())
            score_list.append(tfidf)
        tuple_list = zip(term_list, score_list)
        sorted_tuple_list = sorted(tuple_list, key=lambda x: x[1], reverse=True)
        topk_terms = list(zip(*sorted_tuple_list))[0]
        return topk_terms[:topk]
    
    def getDF(self, term):
        lex = self.index.getLexicon()
        lee = lex.getLexiconEntry(term)
        if lee is not None:
            return lee.getDocumentFrequency()
        else:
            return None
        

    def parse_idf_for_document(self, relevance_dict, topk_term_corpus, rerank_topk=60, doc_topk_term=20):
        idf_map = OrderedDict()
        for qid, relevance in relevance_dict.items():
            supervised_docid_list = relevance.get_supervised_docid_list()[: rerank_topk]
            curr_idf = self.parse_idf_per_query(supervised_docid_list, doc_topk_term, topk_term_corpus)
            idf_map.update({qid: curr_idf})
        
        self.top_idf_dict = idf_map
        pickle.dump(self.top_idf_dict, open(self.topk_idf_file,'wb'))
        return idf_map 


    def parse_idf_per_query(self, supervised_docid_list, doc_topk_term, topk_term_corpus):
        idf_map = OrderedDict()
        for docid in supervised_docid_list:
            doc_topk = topk_term_corpus[docid][1]
            df = np.asarray([self.getDF(t) if self.getDF(t) != None else 1 for t in doc_topk])
            idf = np.log((self.nb_docs - df + 0.5) / (df + 0.5))
            idf_pad = np.zeros((doc_topk_term, ), dtype=np.float32)
            idf_pad[:len(idf)] = idf
            idf_map.update({docid: idf_pad})

        return idf_map
    def sim_mat_and_kernel_d2d(self, relevance_dict, topk_term_corpus, topics, embedding_file, stop_file, kernel_mu_list, kernel_sigma_list, topk_supervised=40, d2d=True):
        embeddings = KeyedVectors.load_word2vec_format(embedding_file, binary=True)
        stoplist = parse_stoplist(stop_file)
        sim_out_dict = OrderedDict()
        
        for qid in tqdm(relevance_dict.keys()):
            
            sim_out = self.sim_mat_and_kernel_per_query(relevance_dict, topk_term_corpus, topics, embeddings, stoplist, 
                                  kernel_mu_list, kernel_sigma_list, topk_supervised, d2d, qid)
            sim_out_dict[qid] = sim_out
            
        return sim_out_dict

    def sim_mat_and_kernel_per_query(self, relevance_dict, topk_term_corpus, topics, embeddings, stoplist, kernel_mu_list, kernel_sigma_list, topk_supervised, d2d,  qid):
        relevance = relevance_dict.get(qid)
        topic_content = topics[topics.qid==qid]['query'].values[0].split()
        supervised_docid_list = relevance.get_supervised_docid_list()
        topk_supervised_docid_list = supervised_docid_list[: topk_supervised]
        # print(topk_supervised_docid_list)
        # sim_mat_dict = OrderedDict()
        # ker_dict = OrderedDict()
        sim_output_dir = make_directory(self.sim_output_path, str(qid))
        ker_output_dir = make_directory(self.kernel_output_path, str(qid))

        OOV_dict = OrderedDict()
        # judged_docid_list = relevance.get_judged_docid_list()
        # cand = judged_docid_list[0] + judged_docid_list[1] + judged_docid_list[2]
        useful_docid_list = supervised_docid_list[: 1000]#[:500] + waitlist
        
        for docid in useful_docid_list:
            sim_mat_list = []
            ker_list = []
            doc_content = self._get_contents(docid)
            if d2d:
                sim_file_name = os.path.join(sim_output_dir, 'q{0}_d{1}.pickle'.format(qid, docid))
            else:
                sim_file_name =  os.path.join(sim_output_dir, 'q{0}_d{1}.npy'.format(qid, docid))
            ker_file_name = os.path.join(ker_output_dir, 'q{0}_d{1}.npy'.format(qid, docid))
            if os.path.exists(sim_file_name):
                continue
            if d2d:
                for sup_docid in topk_supervised_docid_list:
                    sup_doc_content = topk_term_corpus[sup_docid][1][:30]
                    
                    sim_mat = similarity_matrix(sup_doc_content, doc_content, embeddings, OOV_dict)[:, :20000]
                    kernel_feat = kernel_from_matrix(sim_mat, kernel_mu_list, kernel_sigma_list, d2d)
                    sim_mat_list.append(sim_mat.astype(np.float16))
                    ker_list.append(kernel_feat)
                    # if docid==8171 and qid=='1':
                    #     print(sim_mat)
                    #     print(sup_doc_content)
                    #     print(doc_content)
                    #     assert 1 == 2
                    
                    # print(qid, docid, sup_docid)
                    # print(doc_content)
                    # print(sup_doc_content)
                    # assert 1 == 2
                
                ker_list = np.asarray(ker_list)
                # sim_mat_dict[docid] = sim_mat_list
                pickle.dump(sim_mat_list, open(sim_file_name,'wb'))
                np.save(ker_file_name, ker_list)
                
                # ker_dict[docid] = ker_list
            else:
                sim_mat = similarity_matrix(topic_content, doc_content, embeddings, OOV_dict)
                kernel_feat = kernel_from_matrix(sim_mat, kernel_mu_list, kernel_sigma_list, d2d)
                # sim_mat_dict[docid] = sim_mat
                # ker_dict[docid] = kernel_feat
                np.save(sim_file_name, sim_mat)
                np.save(ker_file_name, kernel_feat)
        # print('Finish qid:',qid)
        # return sim_mat_dict, ker_dict

    def hist_d2d(self, relevance_dict, text_max_len, hist_size, sim_out_dict, hist_path, d2d=True):
        
        # hist_dict = OrderedDict()
        # for qid in relevance_dict.keys():
        #     hist_out = self.hist_per_query(relevance_dict, text_max_len, hist_size, sim_out_dict, hist_path, d2d, qid)
        #     hist_dict[qid]=hist_out
        # return hist_dict
        qid_list = relevance_dict.keys()
        # with poolcontext(processes=4) as pool:
        #     pool.map(partial(self.hist_per_query, relevance_dict, text_max_len, hist_size, sim_out_dict, hist_path, d2d), qid_list)
        # print("Finish all!")
        for qid in tqdm(qid_list):
            self.hist_per_query(relevance_dict, text_max_len, hist_size, sim_out_dict, hist_path, d2d, qid)
        print("Finish all!")
 
    def hist_per_query(self, relevance_dict, text_max_len, hist_size, sim_out_dict, hist_path, d2d, qid):
    # relevance_dict = load_pickle(relevance_file)
        hist_output_dir = make_directory(hist_path, str(qid))

        relevance = relevance_dict.get(qid)
        supervised_docid_list = relevance.get_supervised_docid_list()
        # hist_doc_dict = OrderedDict()
        useful_docid_list = supervised_docid_list[: 1000]#[:500] + waitlist
        for docid in useful_docid_list:
            if d2d:
                sim_file_name = os.path.join(self.sim_output_path, str(qid), 'q{0}_d{1}.pickle'.format(qid, docid))
            else:
                sim_file_name = os.path.join(self.sim_output_path, str(qid), 'q{0}_d{1}.npy'.format(qid, docid))
            hist_file_name = os.path.join(hist_output_dir, 'q{0}_d{1}.npy'.format(qid, docid))
            
            if d2d:
                sim_list = pickle.load(open(sim_file_name,'rb'))
                # sim_list = sim_out_dict[qid][0][docid]
                hist_array = np.zeros((len(sim_list), text_max_len, hist_size), dtype=np.float32)
                for i, sim_mat in enumerate(sim_list):
                    sim_mat = sim_mat[:, :20000]
                    hist = hist_from_matrix(text_max_len, hist_size, sim_mat)
                    hist_array[i] = hist

                # hist_doc_dict[docid] = hist_array
                # print(hist_array.shape)
                np.save(hist_file_name, hist_array)
            else:
                # sim_mat = sim_out_dict[qid][0][docid]
                sim_mat = np.load(sim_file_name)
                hist = hist_from_matrix(text_max_len, hist_size, sim_mat)
                # hist_doc_dict[docid] = hist
                np.save(hist_file_name, hist)
        # return hist_doc_dict

    def preprocess(self, res, qrels, topics, top_k_term=5):
        self.relevance_dict = self.create_relevance(res,qrels)
        topk_term_corpus = self.top_k(res, top_k_term)
        self.top_idf_dict = self.parse_idf_for_document(self.relevance_dict,topk_term_corpus, rerank_topk=60, doc_topk_term=top_k_term )
        
        

        # pickle.dump(self.relevance_dict, open(self.relevance_file,'wb'))
        # pickle.dump(self.top_idf_dict, open(self.topk_idf_file,'wb'))

        
        kernel_mu_list = kernal_mus(11, True)
        kernel_sigma_list = kernel_sigmas(11, 0.5, True)
        embedding_file = 'data/GoogleNews-vectors-negative300.bin'
        stop_file = 'data/stopword.txt'
        
        sim_out_dict = self.sim_mat_and_kernel_d2d(self.relevance_dict, topk_term_corpus, topics, embedding_file, stop_file, kernel_mu_list, kernel_sigma_list)
        hist_params = {
        'relevance_dict': self.relevance_dict,
        'text_max_len': 20,
        'hist_size': 10,
        'sim_out_dict': sim_out_dict,
        'hist_path': self.hist_path,
        'd2d': True
        }
        self.hist_d2d(**hist_params)

    def write_result_to_trec_format(self, result_dict, write_path, docnolist_dict):

        
        f = open(write_path, 'w')
        for qid, result in result_dict.items():

            docid_list = result.get_docid_list()
            score_list = result.get_score_list()
            rank = 0
            for docid, score in zip(docid_list, score_list):
                f.write("{0}\tQ0\t{1}\t{2}\t{3}\t{4}\n".format(qid, docnolist_dict[docid], rank, score, result.get_runid()))
                rank += 1

        f.close()

    def eval_by_qid_list(self, X, len_indicator, res_dict, qualified_qid_list,  model,
                       relevance_dict, rerank_topk, nb_supervised_doc, doc_topk_term, qrels_file,
                       docnolist_dict, runid, output_file, ):
    # dd_q, dd_d = list(map(lambda x: x[:, :nb_supervised_doc, : doc_topk_term, :], [dd_q, dd_d]))
        topk_score_all = model.predict_on_batch(X)
        topk_score_all = topk_score_all.flatten()

        for i, qid in enumerate(qualified_qid_list):
            relevance = relevance_dict.get(qid)
            supervised_docid_list = relevance.get_supervised_docid_list()
            topk_score = topk_score_all[sum(len_indicator[:i]): sum(len_indicator[:i]) + len_indicator[i]]

            if len(supervised_docid_list) <= rerank_topk:
                score_list = topk_score
            else:
                behind_score = np.min(topk_score) - 0.001 - np.sort(np.random.random((len(supervised_docid_list) - rerank_topk,)))
                score_list = np.concatenate((topk_score, behind_score))

            res = Result(qid, supervised_docid_list, score_list, runid)
            res.update_ranking()
            res_dict.update({qid: res})
            # print "generate score {0}".format(time.time()-t)
        self.write_result_to_trec_format(res_dict, output_file, docnolist_dict)
        met = evaluate_trec(qrels_file, output_file)

        return met, res_dict

    def _eval_by_qid_list_helper(self, qid_list, pair_generator, nb_supervised_doc, relevance_dict, runid, rerank_topk):
        
        qid_list = sorted(qid_list)
        
        qualified_qid_list = []
        res_dict = OrderedDict()
        for qid in qid_list:
        
            relevance = relevance_dict.get(qid)
            
            supervised_docid_list = relevance.get_supervised_docid_list()
            if len(supervised_docid_list) < nb_supervised_doc:
                # cannot construct d2d feature, thus not need to be update
                score_list = relevance.get_supervised_score_list()
                res = Result(qid, supervised_docid_list, score_list, runid)
                res_dict.update({qid: res})
                # logging.warn("query {0} not to be rerank".format(qid))
            else:
                qualified_qid_list.append(qid)
        # generate re rank score
        dd_q, dd_d, score_gate, len_indicator = \
                            pair_generator.generate_list_batch(qualified_qid_list, rerank_topk)

        return [dd_q, dd_d, score_gate], len_indicator, res_dict, qualified_qid_list

    def _extract_max_metric(self, met):
        index = met[0].index(max(met[0]))
        

        return index, [met[0][index], met[1][index], met[2][index]]

    def fit(self, res, qrels, topics):
        # populate various temporary files that we need, OR
        # just directly populate a PairGenerator.
        
        #tr_qrels: write this out as a qrels file.
        tr_qrels = qrels.copy()
        self.qrels_filename = os.path.join(self.tempdir, "qrels")
        tr_qrels["iteration"] = 0
        tr_qrels[["qid", "iteration", "docno", "label"]].to_csv( self.qrels_filename, sep=" ", index=False, header=False)

        docidlist=res[['docid','docno']].to_numpy()[:,0]
        docnolist=res[['docid','docno']].to_numpy()[:,1]
        docnolist_dict = dict(zip(docidlist, docnolist))

        qid_list = res.qid.unique().tolist()
        random.shuffle(qid_list)
        pat = int(len(qid_list) * 0.8)
        train_qid_list = qid_list[:pat]
        valid_qid_list = qid_list[pat:]

        
        config = NPRFDRMMConfig()
        config.parent_path = self.tempdir
        config.relevance_dict_path = self.relevance_file
        config.dd_q_feature_path = self.topk_idf_file
        config.dd_d_feature_path = self.hist_path
        generator_params = {'relevance_dict_path': self.relevance_file,
                      'dd_q_feature_path': self.topk_idf_file,
                      'dd_d_feature_path': self.hist_path,
                      'sample_perquery_limit': config.sample_perquery_limit,
                      'sample_total_limit': config.sample_total_limit,
                      'query_maxlen': config.query_maxlen,
                      'doc_topk_term': config.doc_topk_term,
                      'nb_supervised_doc': config.nb_supervised_doc,
                      'hist_size': config.hist_size,
                      'batch_size': config.batch_size,
                      'shuffle': True}
        config.regenerateParams(generator_params)
        ddm = NPRFDRMM(config)
        pair_generator = NPRFDRMMPairGenerator(**config.generator_params)
        self.model = ddm.build()
        self.model.compile(optimizer=ddm.config.optimizer, loss=rank_hinge_loss)

        session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        valid_params = ddm.eval_by_qid_list_helper(valid_qid_list, pair_generator)
        nb_pair_train = pair_generator.count_pairs_balanced(train_qid_list, ddm.config.pair_sample_size)

        batch_logger = NBatchLogger(50)
        batch_losses = []
        met = [[], [], [], [], [], []]
        iteration = -1
        best_map = 0.0
        best_p20 = 0.0
        best_ndcg20 = 0.0
        for i in range(ddm.config.nb_epoch):
            print ("Epoch " + str(i))

            nb_batch = nb_pair_train / ddm.config.batch_size

            train_generator = pair_generator.generate_pair_batch(train_qid_list, ddm.config.pair_sample_size)
#             print('nb_batch:', nb_batch)
            for j in range(int(nb_batch / 100)+1):
                print('iteration:', iteration)
                iteration += 1
                history = self.model.fit_generator(generator=train_generator,
                                            steps_per_epoch= nb_pair_train / ddm.config.batch_size,
                                            epochs=1,
                                            shuffle=False,
                                            verbose=0,
                                            callbacks=[batch_logger],
                                            )
                batch_losses.append(batch_logger.losses)
                print("[Iter {0}]\tLoss: {1}".format(iteration, history.history['loss']))

                kwargs = {'model': self.model,
                  'relevance_dict': self.relevance_dict,
                  'rerank_topk': ddm.config.rerank_topk,
                  'qrels_file': self.qrels_filename,
                  'docnolist_dict': docnolist_dict,
                  'runid': ddm.config.runid,
                  'nb_supervised_doc': ddm.config.nb_supervised_doc,
                  'doc_topk_term': ddm.config.doc_topk_term,
                  'output_file': self.output_file}
                # if use_nprf:
                #     kwargs.update({'nb_supervised_doc': ddm.config.nb_supervised_doc,
                #                 'doc_topk_term': ddm.config.doc_topk_term,})

                valid_met, results_dict = self.eval_by_qid_list(*valid_params, **kwargs)
                print("[Valid]\t\tMAP\tP20\tNDCG20")
                print("\t\t{0}\t{1}\t{2}".format(valid_met[0], valid_met[1], valid_met[2]))
                met[0].append(valid_met[0])
                met[1].append(valid_met[1])
                met[2].append(valid_met[2])

                if float(valid_met[0]) > best_map:
                    best_map = float(valid_met[0])
                    best_p20 = float(valid_met[1])
                    best_ndcg20 = float(valid_met[2])
                    self.model.save_weights(self.model_file)


        print('MAP:{}\tP20:{}\tNDCG20:{}'.format(best_map, best_p20, best_ndcg20))
        
        self.model.load_weights(self.model_file)
        self.ddm = ddm

        #if ncessary, write terier's lexicon out to the "dictionary" file

        #write out the topics file.
        # topcis_file = os.path.join(self.tempdir, "topics")
        # tr_topics[["qid", "query"]].write_csv( sep=" ")

        # for i, row in tr_topics.iterrows():
        #     docid = row["docid"]
        #     contents = _get_contents(docid)
        # write out to corpus file


        

        #now call your main data preparation methods. 

        # TODO generate d2d matrix from index?
        #topk_term - this generates the top K terms from each document in index apriori.


        # now call learning
    def transform(self, res, qrels, topics):
        tr_qrels = qrels.copy()
        self.qrels_eval_filename = os.path.join(self.tempdir, "qrels_eval")
        tr_qrels["iteration"] = 0
        tr_qrels[["qid", "iteration", "docno", "label"]].to_csv( self.qrels_eval_filename, sep=" ", index=False, header=False)

        # self.preprocess(res, qrels, topics)
        config = NPRFDRMMConfig()
        config.parent_path = self.tempdir
        config.relevance_dict_path = self.relevance_file
        config.dd_q_feature_path = self.topk_idf_file
        config.dd_d_feature_path = self.hist_path
        generator_params = {'relevance_dict_path': self.relevance_file,
                      'dd_q_feature_path': self.topk_idf_file,
                      'dd_d_feature_path': self.hist_path,
                      'sample_perquery_limit': config.sample_perquery_limit,
                      'sample_total_limit': config.sample_total_limit,
                      'query_maxlen': config.query_maxlen,
                      'doc_topk_term': config.doc_topk_term,
                      'nb_supervised_doc': config.nb_supervised_doc,
                      'hist_size': config.hist_size,
                      'batch_size': config.batch_size,
                      'shuffle': True}
        config.regenerateParams(generator_params)
        
        pair_generator = NPRFDRMMPairGenerator(**config.generator_params)
        test_qid_list = res.qid.unique().tolist()
        test_params = self._eval_by_qid_list_helper(test_qid_list, pair_generator, self.ddm.config.nb_supervised_doc, self.relevance_dict, self.ddm.config.runid, self.ddm.config.rerank_topk)


        docidlist=res[['docid','docno']].to_numpy()[:,0]
        docnolist=res[['docid','docno']].to_numpy()[:,1]
        docnolist_dict = dict(zip(docidlist, docnolist))
        


        kwargs = {'model': self.model,
                  'relevance_dict': self.relevance_dict,
                  'rerank_topk': self.ddm.config.rerank_topk,
                  'qrels_file': self.qrels_eval_filename,
                  'docnolist_dict': docnolist_dict,
                  'runid': self.ddm.config.runid,
                  'nb_supervised_doc': self.ddm.config.nb_supervised_doc,
                  'doc_topk_term': self.ddm.config.doc_topk_term,
                  'output_file': self.output_file}
        test_met, results_dict = self.eval_by_qid_list(*test_params, **kwargs)
        
        print("[Test]\t\tMAP\tP20\tNDCG20")
        print("\t\t{0}\t{1}\t{2}".format(test_met[0], test_met[1], test_met[2]))
        # df_results = pd.DataFrame(results_dict)
        return test_met, results_dict
    # use some of the same steps?? what if documents or terms have been processed before?
