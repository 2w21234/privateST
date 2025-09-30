#!/bin/sh
#SBATCH -J privateST
#SBATCH -p cpu-farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --mem=512G

export OMP_NUM_THREADS=8

module purge
module load gnu12/12.2.0 cuda/12.5


# lmem og
LOG_FILE="memory_log.txt"
echo "=== Memory Tracking Started at $(date) ===" >> $LOG_FILE

# 
(
  while true; do
    echo "[$(date)]" >> $LOG_FILE
    for pid in $(pgrep -u $USER python); do
      if ps -p $pid > /dev/null; then
        rss=$(ps -p $pid -o rss=)
        vsz=$(ps -p $pid -o vsz=)
        rss_gb=$(echo "scale=2; $rss / 1024 / 1024" | bc)
        vsz_gb=$(echo "scale=2; $vsz / 1024 / 1024" | bc)
        cmd=$(ps -p $pid -o args=)
        printf "PID: %s  RSS: %.2f GB  VSZ: %.2f GB  CMD: %s\n" "$pid" "$rss_gb" "$vsz_gb" "$cmd" >> $LOG_FILE
      fi
    done
    echo "-----" >> $LOG_FILE
    sleep 5
  done
) &


python test_privateST.py

exit 0


