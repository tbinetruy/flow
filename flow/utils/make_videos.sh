echo "Making videos..."
ffmpeg -i "$1/frame_%06d.png" -pix_fmt yuv420p all.mp4
ffmpeg -i "$1/sight_rl_0_%06d.png" rl0.mp4
ffmpeg -i "$1/sight_rl_1_%06d.png" rl1.mp4
ffmpeg -i "$1/sight_rl_2_%06d.png" rl2.mp4
ffmpeg -i "$1/sight_rl_3_%06d.png" rl3.mp4
ffmpeg -i rl0.mp4 -i rl1.mp4 -filter_complex vstack rl01.mp4
ffmpeg -i rl2.mp4 -i rl3.mp4 -filter_complex vstack rl23.mp4
ffmpeg -i rl01.mp4 -i rl23.mp4 -filter_complex hstack -pix_fmt yuv420p rl0123.mp4
rm rl0.mp4 rl1.mp4 rl2.mp4 rl3.mp4 rl01.mp4 rl23.mp4
echo "Done."
