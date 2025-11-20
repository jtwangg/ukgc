# from src.dataset.expla_graphs import ExplaGraphsDataset
# from src.dataset.scene_graphs import SceneGraphsDataset
# from src.dataset.scene_graphs_baseline import SceneGraphsBaselineDataset
# from src.dataset.webqsp import WebQSPDataset
# from src.dataset.webqsp_baseline import WebQSPBaselineDataset

# from src.dataset.ppi5k import PPI5kDataset
# from src.dataset.ppi5k_baseline import PPi5kBaselineDataset



from src.dataset.ppi5k_train import PPI5kDataset
from src.dataset.ppi5k_train_baseline import PPi5kBaselineDataset
from src.dataset.nl27k_train import NL27kDataset
from src.dataset.nl27k_train_baseline import NL27kBaselineDataset
from src.dataset.cn15k_train import CN15kDataset
from src.dataset.cn15k_train_baseline import CN15kBaselineDataset

# from src.dataset.cn15k_train_conf import CN15kConfDataset
# from src.dataset.cn15k_train_baseline_conf import CN15kBaselineConfDataset
# from src.dataset.nl27k_train_conf import NL27kConfDataset
# from src.dataset.nl27k_train_baseline_conf import NL27kBaselineConfDataset
# from src.dataset.ppi5k_train_conf import PPI5kConfDataset
# from src.dataset.ppi5k_train_baseline_conf import PPi5kBaselineConfDataset

from src.dataset.nl27k_train_baseline_cp import NL27kBaselineCPDataset
from src.dataset.nl27k_train_cp import NL27kCPDataset
from src.dataset.ppi5k_train_baseline_cp import PPI5kBaselineCPDataset
from src.dataset.ppi5k_train_cp import PPI5kCPDataset
from src.dataset.cn15k_train_baseline_cp import CN15kBaselineCPDataset
from src.dataset.cn15k_train_cp import CN15kCPDataset

from src.dataset.nl27k_train_tc import NL27kTCDataset


load_dataset = {
    # 'expla_graphs': ExplaGraphsDataset,
    # 'scene_graphs': SceneGraphsDataset,
    # 'scene_graphs_baseline': SceneGraphsBaselineDataset,
    # 'webqsp': WebQSPDataset,
    # 'webqsp_baseline': WebQSPBaselineDataset,


    'ppi5k': PPI5kDataset,
    'ppi5k_baseline': PPi5kBaselineDataset,
    'nl27k': NL27kDataset,
    'nl27k_baseline': NL27kBaselineDataset,
    'cn15k': CN15kDataset,
    'cn15k_baseline': CN15kBaselineDataset,
    
    # 'cn15k_conf': CN15kConfDataset,
    # 'cn15k_baseline_conf': CN15kBaselineConfDataset,
    # 'nl27k_conf': NL27kConfDataset,
    # 'nl27k_baseline_conf': NL27kBaselineConfDataset,
    # 'ppi5k_conf': PPI5kConfDataset,
    # 'ppi5k_baseline_conf': PPi5kBaselineConfDataset,

    'nl27k_baseline_cp': NL27kBaselineCPDataset,
    'nl27k_cp': NL27kCPDataset,
    'ppi5k_baseline_cp': PPI5kBaselineCPDataset,
    'ppi5k_cp': PPI5kCPDataset,
    'cn15k_baseline_cp': CN15kBaselineCPDataset,
    'cn15k_cp': CN15kCPDataset,


    'nl27k_tc': NL27kTCDataset,

}
