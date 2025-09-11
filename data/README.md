# Processing EmbodiedScan Data

To use the EmbodiedScan dataset with this project, you first need to download and process the raw data according to the instructions provided by the original EmbodiedScan authors.

## 1. Download and Preprocess Raw Data

Follow the steps outlined in the official EmbodiedScan data preparation guide:
[https://github.com/OpenRobotLab/EmbodiedScan/tree/main/data](https://github.com/OpenRobotLab/EmbodiedScan/tree/main/data)

An example of the steps is as follows:
1. Download ScanNet v2 data.  
2. Download 3RScan data.  
3. Download Matterport3D data.  
4. Download ARKitScenes data.  
5. Download EmbodiedScan annotations (`.pkl` files).  
6. Extract images for ScanNet and 3RScan using the provided scripts (`generate_image_scannet.py` and `generate_image_3rscan.py`).  

**Note:** You do **not** need to perform the final step in the EmbodiedScan README (extracting occupancy annotations) for this project.


Ensure your final data directory structure matches the one specified in the EmbodiedScan README.

## 2. Update Configuration File

Once the data is downloaded and processed, you need to update the configuration file to point to the correct locations on your system.

Edit the `robospatial/configs/embodiedscan.yaml` file.

Update the following paths under the `data_loading` section:
-   `image_root`: Set this to the directory where the extracted images (e.g., `scannet/posed_images`, `3rscan/<scene_id>/sequence`) are located. The specific structure might depend on how you organized the datasets downloaded in step 1.
-   `embodiedscan_ann`: Update the `train`, `val`, and `test` paths to point to the downloaded `.pkl` annotation files (from step 5).

Example relevant section in `robospatial/configs/embodiedscan.yaml`:

```yaml
data_loading:
  # ... other settings ...
  image_root: /path/to/your/processed/image/data # <- UPDATE THIS
  embodiedscan_ann:
    train: /path/to/your/EmbodiedScan/data/embodiedscan_infos_train.pkl # <- UPDATE THIS
    val:   /path/to/your/EmbodiedScan/data/embodiedscan_infos_val.pkl   # <- UPDATE THIS
    test:  /path/to/your/EmbodiedScan/data/embodiedscan_infos_test.pkl  # <- UPDATE THIS
  # ... other settings ...
```

After completing these steps, you should be able to load and use the EmbodiedScan dataset with the project.
