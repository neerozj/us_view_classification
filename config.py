class Config(object):
  """For Training and validation"""
  depth = 10
  num_classes = 3
  num_epoch = 500
  lr_start = 1e-3
  lr_min = 1e-3
  augment=False

  num_views = 1
  batch_size = 10
  eval_batch_size = 1
  test_batch_size = 1
  image_size = 120
  num_frames = 10
  num_channels = 1
  classes = ['Ap4',
             'Ap2',
             'Others']
  augment_ratio = 5
  weight_decay=1e-4

  """For Testing"""
  src_test_folder = '/media/truecrypt1/EF_Estimation_Round3/EF_Estimation_509cases_2017.5.10/MatAnon'
  dest_test_folder = '/media/neeraj/pdf/cardiac_dys/EF_Estimation_509cases_2017.5.10_ViewClassified'
  model_folder = 'best_model'
  start_idx = 0 #change this if the process is interrupted because of the memory issue and start from the cases that you wnat
  cpu = True #by default it runs test inference on CPU with no disturbance










