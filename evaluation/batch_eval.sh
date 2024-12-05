# example
EXP_DIR="/mnt/nfs/data/guangcong/DynamiCrafter/test_results/test_256_cami2v_pluckerEmbedding+EpipolarAttn_randCondFrame/images/test/ImageTextcfg7.5_CameraConditionTrue_CameraCfg1.0_eta1.0_guidanceRescale0.7_cfgScheduler=constant_steps25"

for trial_id in 0 1 2 3 4; do
    echo $trial_id

    CUDA_VISIBLE_DEVICES=$trial_id python glomap_evaluation.py --exp_dir $EXP_DIR --trial_id $trial_id &
done

wait

python utils/merge.py
python utils/summary.py

python fvd_test.py --gt_folder $EXP_DIR/gt_video --sample_folder $EXP_DIR/samples