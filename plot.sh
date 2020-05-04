
FONTSCALE=1
INIT_GNUPLOT="set terminal wxt lw $FONTSCALE; set terminal wxt fontscale $FONTSCALE"

function age() {
   local filename=$1
   local changed=`stat -c %Y "$filename"`
   local now=`date +%s`
   local elapsed

   let elapsed=now-changed
   echo $elapsed
}

gnuplot -e "$INIT_GNUPLOT; set key autotitle columnhead; plot for [col=2:2] 'episodeLength' using 0:col with points; pause mouse close; " &
gnuplot -e "$INIT_GNUPLOT; set key autotitle columnhead; plot for [col=2:4] 'stateValues' using 0:col with lines; pause mouse close; " &
gnuplot -e "$INIT_GNUPLOT; set key autotitle columnhead; plot for [col=5:6] 'stateValues' using 0:col with points; pause mouse close; " &
gnuplot -e "$INIT_GNUPLOT; set key autotitle columnhead; plot for [col=2:3] 'costs' using 0:col with points; pause mouse close; " &
gnuplot -e "$INIT_GNUPLOT; set key autotitle columnhead; plot for [col=2:2] 'reward' using 0:col with points; pause mouse close; " &

if [[ $(age "$file") < 300 ]];
then
    NR="`head -n1 stateValuesAllStatesCount`"
    gnuplot -e "$INIT_GNUPLOT; set key autotitle columnhead; plot for [col=2:$((NR+1))] 'stateVAllStates' using 1:col with lines; set key title 'All V Values'; pause mouse close; " &
    gnuplot -e "$INIT_GNUPLOT; set key autotitle columnhead; plot for [col=2:$((NR+1))] 'stateWAllStates' using 1:col with lines; set key title 'All W Values'; pause mouse close; " &
    gnuplot -e "$INIT_GNUPLOT; set key autotitle columnhead; plot for [col=2:$((NR+1))] 'statePsiVAllStates' using 1:col with lines; set key title 'All Psi V Values'; pause mouse close; " &
    gnuplot -e "$INIT_GNUPLOT; set key autotitle columnhead; plot for [col=2:$((NR+1))] 'statePsiWAllStates' using 1:col with lines; set key title 'All Psi W Values'; pause mouse close; " &
fi


# set term wxt 0
# plot for [col=2:2] 'episodeLength' using 0:col with points
# set term wxt 1
# plot for [col=2:6] 'stateValues' using 0:col with lines
# pause mouse close


# watch 'tail -n 10000 queueLength | awk "{ sum += \$1; n++ } END { if (n > 0) print sum / n ; }"'
# watch 'tail -n 1000 episodeLength | awk "{ sum += \$2; n++ } END { if (n > 0) print sum / n ; }"'
# watch 'pr -m -t reward costs | tail -n 1000 - | awk "{ sum += \$2; sum4 += \$4; n++ } END { if (n > 0) print (sum / n, sum4 /n) ; }"; pr -m -t reward costs | tail -n 10000 - | awk "{ sum += \$2; sum4 += \$4; n++ } END { if (n > 0) print (sum / n, sum4 /n) ; }"; pr -m -t reward costs | tail -n 100000 - | awk "{ sum += \$2; sum4 += \$4; n++ } END { if (n > 0) print (sum / n, sum4 /n) ; }"'
