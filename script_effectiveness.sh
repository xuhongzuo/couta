flag=.effectivenss

for data in ASD SMD WaQ SWaT Epilepsy DSADS;do
# for data in SMD;do
    echo $data
    for algo in COUTA COUTA_wto_nac COUTA_wto_umc Canonical;do
        python -u main.py --data $data --algo $algo --runs 5 --flag $flag > /dev/null &
    done
    wait
done
