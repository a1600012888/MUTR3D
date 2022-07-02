import mmcv
import os
import torch
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
import time
from mmdet3d.apis import show_result_meshlab

def single_gpu_visualize(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    print('begin single gpu visualization!')
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result_tmp = model(return_loss=False, **data)
        new_result_tmp = []
        for _result_tmp in result_tmp:
            if 'pts_bbox' in _result_tmp:
                new_result_tmp.append(_result_tmp['pts_bbox'])
        
        results.extend(new_result_tmp)
        if i > 100:
            break
        batch_size = len(result_tmp)
        for _ in range(batch_size):
            prog_bar.update()

    return results


def multi_gpu_visualize(model, data_loader, out_dir=None,
                    show_score_thr=0.3):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    print('begin single gpu visualization!')
    model.eval()
    results = []
    dataset = data_loader.dataset
    
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result_tmp = model(return_loss=False, **data)
        if i > 5000:
            break
        
        new_result_tmp = []
        for _result_tmp in result_tmp:
            if 'pts_bbox' in _result_tmp:
                new_result_tmp.append(_result_tmp['pts_bbox'])
        
        results.extend(new_result_tmp)

        batch_size = len(result_tmp)
        for _ in range(batch_size):
            prog_bar.update()

    # collect results from all ranks
    
    return results
