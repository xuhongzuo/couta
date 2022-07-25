data_root=data_processed_contam/
flag=.robustness.test

for r in 0.0 0.04 0.08 0.12 0.16;do
    data=DSADS_contam_$r
    echo $data
    python -u main.py --data_dir $data_root --results_dir @results_robust/ --data $data --entities FULL --algo COUTA --runs 5 --flag $flag > /dev/null &
done
wait

for r in 0.0 0.04 0.08 0.12 0.16 0.2 0.24;do
    data=EP_contam_$r
    echo $data
    python -u main.py --data_dir $data_root --results_dir @results_robust/ --data $data --entities FULL --algo COUTA --runs 5 --flag $flag > /dev/null &
done
wait