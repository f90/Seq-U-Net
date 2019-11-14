# DOWNLOAD
youtube-dl --ignore-errors -a dataset_test.txt -f bestaudio -x --audio-format mp3 -o 'test/%(id)s_%(duration)s_%(title)s.%(ext)s'
youtube-dl --ignore-errors -a dataset_train.txt -f bestaudio -x --audio-format mp3 -o 'train/%(id)s_%(duration)s_%(title)s.%(ext)s'

# SPLIT FILES INTO 30min PIECES SO AUDIO LOADING DOES NOT BREAK
for filename in test/*.mp3; do
    ffmpeg -i "$filename" -f segment -segment_time 1800 -c copy "$filename".%03d.mp3
    rm "$filename"
done

for filename in train/*.mp3; do
    ffmpeg -i "$filename" -f segment -segment_time 1800 -c copy "$filename".%03d.mp3
    rm "$filename"
done
