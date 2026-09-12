"""Write the nominal AS2 UMI configuration without launching a simulator."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pawcerto.methods.umi_on_legs.robot_binding import as2_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'reference/checkpoints/tossing/ours/config.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=as2_config(json.loads(args.config.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps(dict(config=str(args.output.resolve()),robot=config['pawcerto_robot'],
        actor_obs=config['runner']['alg']['actor_critic']['num_actor_obs'],
        critic_obs=config['runner']['alg']['actor_critic']['num_critic_obs']),indent=2))


if __name__=='__main__':
    main()
