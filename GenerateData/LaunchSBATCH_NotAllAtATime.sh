# Maximum number of jobs
MAX=2

# Loop over the job IDs
for JobNum in {1..11}
do
    # Number of processes in the sbatch queue
    NS=$(squeue | grep rtoma)
    NS=$(echo "$NS" | wc -l)

    # If too much jobs in the queue, just wait before launching other jobs
    while [ "$NS" -gt "$MAX" ]
    do
        # Sleep for 30 seconds, and check the number of jobs in the sbatch queue again
        sleep 30
        NS=$(squeue | grep rtoma)
        NS=$(echo "$NS" | wc -l)
        echo "Number of slurm tasks is $NS (max $MAX)..."
    done
    
    # One slot can be used for a new slurm job
    sbatch --export=JOBID=$JobNum --job-name=GenerateData_IncreasingSizewhatIsTheLimit$JobNum --output=GenerateData_IncreasingSizewhatIsTheLimit.txt$JobNum Launch_StudySpeed_EmbarassinglyParallel.sh
    echo "Job $JobNum has been launched !"
done
