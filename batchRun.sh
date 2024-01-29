nohup python run.py --agent fixedtime >> manhattan7x7-final/fixedtime_no_mal.nohup.out 2>&1 &
nohup python run.py --agent sotl >> manhattan7x7-final/sotl_no_mal.nohup.out 2>&1 &
nohup python run.py --agent maxpressure >> manhattan7x7-final/maxpressure_no_mal.nohup.out 2>&1 &
nohup python run.py --ngpu 0 --agent dqn >> manhattan7x7-final/dqn_no_mal.nohup.out 2>&1 &
nohup python run.py --ngpu 0 --agent presslight >> manhattan7x7-final/presslight_no_mal.nohup.out 2>&1 &
nohup python run.py --ngpu 0 --agent prlight >> manhattan7x7-final/colight_no_mal.nohup.out 2>&1 &


nohup python run.py --agent fixedtime >> manhattan7x7-final/fixedtime_mal.nohup.out 2>&1 &
nohup python run.py --agent sotl >> manhattan7x7-final/sotl_mal.nohup.out 2>&1 &
nohup python run.py --agent maxpressure >> manhattan7x7-final/maxpressure_mal.nohup.out 2>&1 &
nohup python run.py --ngpu 1 --agent dqn >> manhattan7x7-final/dqn_mal.nohup.out 2>&1 &
nohup python run.py --ngpu 1 --agent presslight >> manhattan7x7-final/presslight_mal.nohup.out 2>&1 &
nohup python run.py --ngpu 1 --agent prlight >> manhattan7x7-final/colight_mal.nohup.out 2>&1 &
