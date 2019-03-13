
gnuplot -e "set key autotitle columnhead; plot for [col=2:2] 'episodeLength' using 0:col with points; pause mouse close; " &
gnuplot -e "set key autotitle columnhead; plot for [col=2:4] 'stateValues' using 0:col with lines; pause mouse close; " &
gnuplot -e "set key autotitle columnhead; plot for [col=5:6] 'stateValues' using 0:col with lines; pause mouse close; " &

# set term wxt 0
# plot for [col=2:2] 'episodeLength' using 0:col with points
# set term wxt 1
# plot for [col=2:6] 'stateValues' using 0:col with lines
# pause mouse close
