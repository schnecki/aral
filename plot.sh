
gnuplot -e "set key autotitle columnhead; plot for [col=2:2] 'episodeLength' using 0:col with points; pause mouse close; " &
gnuplot -e "set key autotitle columnhead; plot for [col=2:4] 'stateValues' using 0:col with lines; pause mouse close; " &
gnuplot -e "set key autotitle columnhead; plot for [col=5:6] 'stateValues' using 0:col with lines; pause mouse close; " &
gnuplot -e "set key autotitle columnhead; plot for [col=2:3] 'costs' using 0:col with points; pause mouse close; " &
gnuplot -e "set key autotitle columnhead; plot for [col=2:2] 'reward' using 0:col with points; pause mouse close; " &

NR="`head -n1 stateValuesAllStatesCount`"
if [ $? -eq 0 ]; then
    gnuplot -e "set key autotitle columnhead; plot for [col=2:$((NR+1))] 'stateValuesAllStates' using 1:col with lines; set key title 'All Bias Values'; pause mouse close; " &
    gnuplot -e "set key autotitle columnhead; plot for [col=2:$((NR+1))] 'statePsiVAllStates' using 1:col with lines; set key title 'All Psi V Values'; pause mouse close; " &
    gnuplot -e "set key autotitle columnhead; plot for [col=2:$((NR+1))] 'statePsiWAllStates' using 1:col with lines; set key title 'All Psi W Values'; pause mouse close; " &
    gnuplot -e "set key autotitle columnhead; plot for [col=2:$((NR+1))] 'statePsiW2AllStates' using 1:col with lines; set key title 'All Psi W2 Values'; pause mouse close; " &
fi

# set term wxt 0
# plot for [col=2:2] 'episodeLength' using 0:col with points
# set term wxt 1
# plot for [col=2:6] 'stateValues' using 0:col with lines
# pause mouse close


# watch 'tail -n 10000 queueLength | awk "{ sum += \$1; n++ } END { if (n > 0) print sum / n ; }"'
