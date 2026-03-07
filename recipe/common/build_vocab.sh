# NOTE: the script is supposed to be used in recipes like:
#     . script.sh
# Please don't try to run the shell directly.

uv run task build-vocab -- $feature_file_dir $vocab_dir -m $vocab_min_freq
