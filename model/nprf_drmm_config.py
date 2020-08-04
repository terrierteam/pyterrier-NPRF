import os
import numpy as np
from utils.file_operation import make_directory

class NPRFDRMMConfig():

  # qxd config
  # input config
  query_maxlen = 5
  hist_size = 10

  # model config
  nb_layers = 1
  hidden_size = [5]
  out_size = 1

  # dxd config
  nb_supervised_doc = 20 #10
  doc_topk_term = 20

  # merge_config
  merge_hidden = 5
  merge_out = 1

  # training config
  loss_func = "rank_hinge_loss"
  optimizer = "adam"
  batch_size = 20
  nb_epoch = 500
  max_iteration = 250
  learning_rate = 0.001
  pair_sample_size = 10
  sample_perquery_limit = 1000
  sample_total_limit = 10000

  # rerank config
  rerank_topk = 500
  nb_docid_per_query = 1000
  method = 3
  ''' method | doc gating | dense | name 
        1         No         Yes     ff'
        2        Yes         Yes     ff
        3        Yes          No     ds
  '''

  # generator config
  parent_path = '../data/'
  relevance_dict_path = os.path.join(parent_path, 'relevance.txt')

  dd_q_feature_path = os.path.join(parent_path, 'topk.idf.pkl')
  dd_d_feature_path = os.path.join(parent_path, 'histogram')

  generator_params = {'relevance_dict_path': relevance_dict_path,
                      'dd_q_feature_path': dd_q_feature_path,
                      'dd_d_feature_path': dd_d_feature_path,
                      'sample_perquery_limit': sample_perquery_limit,
                      'sample_total_limit': sample_total_limit,
                      'query_maxlen': query_maxlen,
                      'doc_topk_term': doc_topk_term,
                      'nb_supervised_doc': nb_supervised_doc,
                      'hist_size': hist_size,
                      'batch_size': batch_size,
                      'shuffle': True}
  

  # qid config
  qid_list_dict = {
    'disk00':[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 
    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38], 
    [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57], 
    [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76], 
    [77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93]],

    'disk12': [
      [56, 57, 71, 73, 77, 88, 93, 100, 102, 103, 104, 109, 115, 121, 123, 124, 128, 134, 135, 138, 149, 152, 157, 160,
       169, 177, 178, 183, 194, 198],
      [61, 63, 67, 68, 78, 79, 90, 101, 106, 110, 113, 119, 120, 127, 129, 136, 141, 154, 159, 161, 172, 173, 175, 176,
       181, 186, 187, 193, 195, 197],
      [51, 58, 60, 70, 82, 84, 85, 86, 87, 89, 91, 92, 94, 96, 98, 105, 107, 111, 117, 133, 137, 146, 156, 158, 163,
       166, 174, 179, 190, 196],
      [52, 54, 62, 64, 66, 72, 74, 76, 80, 83, 99, 125, 126, 143, 144, 147, 148, 151, 155, 167, 168, 170, 171, 180, 184,
       185, 188, 189, 191, 200],
      [53, 55, 59, 65, 69, 75, 81, 95, 97, 108, 112, 114, 116, 118, 122, 130, 131, 132, 139, 140, 142, 145, 150, 153,
       162, 164, 165, 182, 192, 199]],

    'disk45': [
      [302, 303, 309, 316, 317, 319, 323, 331, 336, 341, 356, 357, 370, 373, 378, 381, 383, 392, 394, 406, 410, 411,
       414, 426, 428, 433, 447, 448, 601, 607, 608, 612, 617, 619, 635, 641, 642, 646, 647, 654, 656, 662, 665, 669,
       670, 679, 684, 690, 692, 700],
      [301, 308, 312, 322, 327, 328, 338, 343, 348, 349, 352, 360, 364, 365, 369, 371, 374, 386, 390, 397, 403, 419,
       422, 423, 424, 432, 434, 440, 446, 602, 604, 611, 623, 624, 627, 632, 638, 643, 651, 652, 663, 674, 675, 678,
       680, 683, 688, 689, 695, 698],
      [306, 307, 313, 321, 324, 326, 334, 347, 351, 354, 358, 361, 362, 363, 376, 380, 382, 396, 404, 413, 415, 417,
       427, 436, 437, 439, 444, 445, 449, 450, 603, 605, 606, 614, 620, 622, 626, 628, 631, 637, 644, 648, 661, 664,
       666, 671, 677, 685, 687, 693],
      [320, 325, 330, 332, 335, 337, 342, 344, 350, 355, 368, 377, 379, 387, 393, 398, 402, 405, 407, 408, 412, 420,
       421, 425, 430, 431, 435, 438, 616, 618, 625, 630, 633, 636, 639, 649, 650, 653, 655, 657, 659, 667, 668, 673,
       676, 682, 686, 691, 697],
      [304, 305, 310, 311, 314, 315, 318, 329, 333, 339, 340, 345, 346, 353, 359, 366, 367, 372, 375, 384, 385, 388,
       389, 391, 395, 399, 400, 401, 409, 416, 418, 429, 441, 442, 443, 609, 610, 613, 615, 621, 629, 634, 640, 645,
       658, 660, 681, 694, 696, 699]]
  }
  runid = "NPRF-DRMM-BM25NOQE.method{0}.ED{1}.ET{2}".format(method, nb_supervised_doc, doc_topk_term)

  def regenerateParams(self, generator_params):
    self.generator_params = generator_params

  # qid_list = qid_list_dict['disk00']

  # # evaluate config
  # docnolist_file = os.path.join(parent_path, 'docno.txt')
  # qrels_file = os.path.join(parent_path, 'qrels')

  # # save config
  # save_path = os.path.join(parent_path, 'model.weight/NPRF.DRMM')
  # 
  # result_path = make_directory(os.path.join(parent_path, 'result/NPRF.DRMM'), runid)
