#!/bin/sh
#SBATCH -J resnet_18_BrSTNET
#SBATCH -p cpu-farm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --mem=490G

export OMP_NUM_THREADS=8

module purge
module load gnu12/12.2.0 cuda/12.5


# 로그 파일
LOG_FILE="20250922_512_log.txt"
echo "=== Memory Tracking Started at $(date) ===" >> $LOG_FILE

# 백그라운드 메모리 로깅 함수 실행
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

# 실제 python 스크립트 실행

#python run_resnet_0425_test_12.py

#python Br_STNet_baseline_best10_0520.py  --gene_filter 10 --count_root training/counts/512/Breast_cancer/   --img_root training/images/512/Breast_cancer/   --test_count_root test/counts/512/Breast_cancer/   --test_img_root test/images/512/Breast_cancer/   --test_patient_file test_patients.csv   --resolution 64 --model resnet18 --pretrained --aux_ratio 0 --cv_fold 5
#python test_fhe.py
#python test_fhe_train_loader.py
python test_fhe_train_loader_memory.py
#python Br_STNet_baseline_best10_0520.py  --gene_filter 10 --count_root training/counts/512/Breast_cancer/   --img_root training/images/512/Breast_cancer/   --test_count_root test/counts/512/Breast_cancer/   --test_img_root test/images/512/Breast_cancer/   --test_patient_file test_patients.csv   --resolution 64 --model resnet18 --aux_ratio 0 --cv_fold 2 --cv_epochs 1 --epochs 1


#python Br_STNet_baseline_best10_0520.py --cv_epochs 1 --epochs 1 --gene_filter 10 --count_root training/counts/512/Breast_cancer/   --img_root training/images/512/Breast_cancer/   --test_count_root test/counts/512/Breast_cancer/   --test_img_root test/images/512/Breast_cancer/   --test_patient_file test_patients.csv   --resolution 64 --model resnet18 --pretrained --aux_ratio 0 --cv_fold 2


exit 0


