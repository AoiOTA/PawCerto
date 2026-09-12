"""UniFP runner with complete configuration and force-stage continuation."""
import json
from pathlib import Path
import torch
from ..algorithm.on_policy_runner import OnPolicyRunner as OriginalRunner


class OnPolicyRunner(OriginalRunner):
    def save(self, path, infos=None):
        torch.save({'model_state_dict':self.alg.actor_critic.state_dict(),
                    'optimizer_state_dict':self.alg.optimizer.state_dict(),
                    'adaptation_optimizer_state_dict':self.alg.adaptation_module_optimizer.state_dict(),
                    'learning_rate':self.alg.learning_rate,'iter':self.current_learning_iteration,
                    'global_steps':self.env.global_steps,'config':self.env.config,'infos':infos,
                    'resume_semantics':'optimizer/iteration/force-stage continuation; fresh physical episodes'},path)

    def load(self, path, load_optimizer=True):
        saved = torch.load(path,map_location='cpu',weights_only=False)
        if 'global_steps' not in saved or 'config' not in saved:
            raise ValueError('Training resume requires resolved config and actual global_steps')
        result = super().load(path,load_optimizer=load_optimizer)
        self.env.global_steps = int(saved['global_steps'])
        return result

    def log(self, locs, width=80, pad=35):
        super().log(locs,width,pad)
        metrics = {'iteration':locs['it']+1,'global_steps':self.env.global_steps,
            'value_loss':locs['mean_value_loss'],'surrogate_loss':locs['mean_surrogate_loss'],
            'adaptation_loss':locs['mean_adaptation_module_loss'],
            'learning_rate':self.alg.learning_rate,'force_stage':self.env.force_stage_info,
            'collection_seconds':locs['collection_time'],'learning_seconds':locs['learn_time']}
        with (Path(self.log_dir)/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(metrics,allow_nan=False)+'\n')
        print('UNIFP_METRICS '+json.dumps(metrics,allow_nan=False),flush=True)
