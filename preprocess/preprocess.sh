
task(){
    python preprocess/read_bags.py $d True
    python preprocess/process_uwb.py $d
    python preprocess/cleanup_csv.py $d
}

N=16
(
for d in data/*/ ; do
    if [ ${#d} -gt 9 ]; then
        continue
    fi
    ((i=i%N)); ((i++==0)) && wait
    task $d &
done
)