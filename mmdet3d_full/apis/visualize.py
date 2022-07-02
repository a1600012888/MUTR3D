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
    results_all_levels = [[], [], [], [], [], []]
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result_all_levels = model(return_loss=False, rescale=True, visualize=True, **data)
        if i > 100:
            break
        for j, result in enumerate(result_all_levels):
            if show:
                # Visualize the results of MMdetection3D model
                # 'show_results' is MMdetection3D visualization API
                if hasattr(model.module, 'show_results'):
                    model.module.show_results(data, result, out_dir+'-level-{}'.format(j))
                # Visualize the results of MMdetection model
                # 'show_result' is MMdetection visualization API
                else:
                    batch_size = len(result)
                    if batch_size == 1 and isinstance(data['img'][0],
                                                    torch.Tensor):
                        img_tensor = data['img'][0]
                    else:
                        img_tensor = data['img'][0].data[0]
                    img_metas = data['img_metas'][0].data[0]
                    imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                    assert len(imgs) == len(img_metas)

                    for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                        h, w, _ = img_meta['img_shape']
                        img_show = img[:h, :w, :]

                        ori_h, ori_w = img_meta['ori_shape'][:-1]
                        img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                        if out_dir:
                            out_file = os.path.join(out_dir,
                                                    '{}-level-'.format(j) + img_meta['ori_filename'])
                        else:
                            out_file = None

                        model.module.show_result(
                            img_show,
                            result[i],
                            show=show,
                            out_file=out_file,
                            score_thr=show_score_thr)
            results_all_levels[j].extend(result)

        
        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    return results_all_levels


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
    results_all_levels = [[], [], [], [], [], []]
    dataset = data_loader.dataset
    
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result_all_levels = model(return_loss=False, rescale=True, visualize=True, **data)
        if i > 5000:
            break
        for j, result in enumerate(result_all_levels):
            if rank == 0:
                # Visualize the results of MMdetection3D model
                # 'show_results' is MMdetection3D visualization API
                if hasattr(model.module, 'show_results'):
                    visual_data = {}
                    visual_data['img_metas'] = data['img_metas'][0].data
                    visual_data['points'] = data['points'][0].data
                    
                    #show_result_meshlab(visual_data, result, 
                    #    out_dir+'-level-{}/smaple-{}'.format(j, i),
                    #    score_thr=0.3,)
                    #model.module.show_results(data, result, out_dir+'-level-{}'.format(j))
                # Visualize the results of MMdetection model
                # 'show_result' is MMdetection visualization API
                
            results_all_levels[j].extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    # collect results from all ranks
    
    results_all_levels = dataset._format_bbox_all_levels(results_all_levels)
    return results_all_levels
