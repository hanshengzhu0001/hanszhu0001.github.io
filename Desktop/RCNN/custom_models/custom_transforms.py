from mmdet.datasets.transforms import PackDetInputs
from mmdet.registry import TRANSFORMS

@TRANSFORMS.register_module()
class PackChartInputs(PackDetInputs):
    """Custom transform to pack chart metadata."""
    
    def transform(self, results: dict) -> dict:
        """Transform function to pack chart metadata.
        
        Args:
            results (dict): Result dict from the data pipeline.
            
        Returns:
            dict: The dict contains the data wrapped with ``DataSample``.
        """
        print(f"DEBUG: PackChartInputs received results: {results.keys()}")
        
        # Call parent transform first
        data_sample = super().transform(results)
        
        # Add chart metadata
        metadata = {}
        for key in ['chart_type', 'plot_bb', 'axes_info', 'data_series']:
            if key in results:
                metadata[key] = results[key]
                print(f"DEBUG: Found metadata key {key}: {results[key]}")
        
        if metadata:
            print(f"DEBUG: Setting metadata: {metadata}")
            # Update existing metainfo instead of replacing it
            current_meta = data_sample.metainfo
            current_meta.update(metadata)
            data_sample.set_metainfo(current_meta)
            
        return data_sample 