
import os
import tqdm
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import open_slide


def tiling(_label_file, _test_file=None, _tile_size=2240, _tile_level=5):

    _train_tile_base = "train_output_png_path"
    if not os.path.exists(_train_tile_base):
        os.mkdir(_train_tile_base)
    if _test_file is not None:
        _test_tile_base = "test_output_png_path"
        if not os.path.exists(_test_tile_base):
            os.mkdir(_test_tile_base)
    else:
        print("No testing WSI is provided.")

    _format = "png"
    _tile_size = _tile_size-2 
    _overlap = 1  
    _limit_bounds = True
    _quality = 90  
    _workers = 6

    if _label_file is not None:
        print("Training WSIs: start tiling ...")
        _label_df = pd.read_csv(_label_file)
        _svs_path = _label_df.iloc[:, 0]
        _svs_label = _label_df.iloc[:, 1]  
        for i in tqdm.tqdm(range(len(_svs_path))):
            _curr_svs = _svs_path.iloc[i]  
            _folder_name = os.path.join(_train_tile_base, '\\'.join(_curr_svs.split("\\")[-2:]).split(".")[0])   # path in windows
            print(_folder_name)
            open_slide.DeepZoomStaticTiler(_curr_svs, _folder_name, _format,
                                           _tile_size, _overlap, _limit_bounds, _quality,
                                           _workers, _tile_level).run()
            print("one slide file is done")
        print("The number of train slide is:", len(_svs_path))
        print("All train slide are done")
    else:
        print("No training WSI is provided.")

    if _test_file is not None:
        print("Testing WSIs: start tiling ...")
        _label_df = pd.read_csv(_test_file)
        _svs_path = _label_df.iloc[:, 0]
        _svs_label = _label_df.iloc[:, 1]

        for i in tqdm.tqdm(range(len(_svs_path))):
            _curr_svs = _svs_path.iloc[i]
            _folder_name = os.path.join(_test_tile_base, '\\'.join(_curr_svs.split("\\")[-2:]).split(".")[0])
            open_slide.DeepZoomStaticTiler(_curr_svs, _folder_name, _format,
                                           _tile_size, _overlap, _limit_bounds, _quality,
                                           _workers, _tile_level).run()

        print("The number of test slide is:", len(_svs_path))
        print("All test slide are done")

    else:
        print("No testing WSI is provided.")

if __name__ == '__main__':
        tiling('path/slide_path.csv', _test_file=None,     #the pait of slide_path and label in csv
                _tile_size=2240, _tile_level=5)  #_tile_level is objective magnification
 