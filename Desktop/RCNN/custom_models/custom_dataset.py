from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS
import json
import os

@DATASETS.register_module()
class ChartDataset(CocoDataset):
    """Custom dataset for chart detection with metadata."""
    
    def load_data_list(self):
        """Load annotations and add metadata to data samples."""
        data_list = super().load_data_list()
        
        # Load annotations file
        with open(self.ann_file, 'r') as f:
            annotations = json.load(f)
            
        # Create filename to metadata mapping
        metadata_map = {}
        for img_info in annotations['images']:
            filename = img_info['file_name']
            metadata_map[filename] = {
                'chart_type': img_info.get('chart_type', 'Unknown'),
                'plot_bb': img_info.get('plot_bb', None),
                'axes_info': img_info.get('axes_info', None),
                'data_series': img_info.get('data_series', None)
            }
            print(f"DEBUG: Loaded metadata for image {filename}: {metadata_map[filename]}")
        
        # Add metadata to each data sample
        for data_info in data_list:
            img_path = data_info['img_path']
            filename = os.path.basename(img_path)
            if filename in metadata_map:
                data_info.update(metadata_map[filename])
                print(f"DEBUG: Added metadata to data_info for image {filename}: {data_info}")
                
        return data_list 