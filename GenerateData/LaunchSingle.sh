#!/bin/bash
#
#SBATCH --job-name=GenerateData_IncreasingSizewhatIsTheLimit
#SBATCH --output=GenerateData_IncreasingSizewhatIsTheLimit.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=60:00
#SBATCH --mem-per-cpu=10000


FILES="SamplingParameterSpaceTestSpeed_1"

module load Python/3.6.3-foss-2017b

mpirun python3 /home/ulg/costmo/rtoma/3DPrinting/GenerateData/GenerateData_IncreasingSizewhatIsTheLimit.py $FILES
