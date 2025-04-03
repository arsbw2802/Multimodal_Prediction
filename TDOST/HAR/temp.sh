

for file in /mnt/attached1/TDOST/HAR/code/saved_logs/kyoto7/*wd_0_bs*; do
  mv "$file" "${file/wd_0_bs/wd_0.0_bs}"
done
