SPLIT=$1

python datasets/utils/generate_dataset.py --split $SPLIT
python datasets/utils/gather_realestate.py --split $SPLIT
python datasets/utils/get_realestate_clips.py --split $SPLIT
python datasets/utils/preprocess_realestate.py --split $SPLIT