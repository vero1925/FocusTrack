from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.focustrack.config import cfg, update_config_from_file


def parameters(yaml_name: str, test_epoch=None):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/focustrack/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    # print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE.FACTOR
    params.template_size = cfg.TEST.TEMPLATE.SIZE
    params.search_factor = cfg.TEST.SEARCH.FACTOR
    params.search_size = cfg.TEST.SEARCH.SIZE

    # Network checkpoint path
    if test_epoch is None:
        params.checkpoint = os.path.join(save_dir, "checkpoints/train/focustrack/%s/FocusTrack_ep%04d.pth.tar" %
                                     (yaml_name, cfg.TEST.EPOCH))
    else:
        params.checkpoint = os.path.join(save_dir, "checkpoints/train/focustrack/%s/FocusTrack_ep%04d.pth.tar" %
                                     (yaml_name, test_epoch))
    
    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
