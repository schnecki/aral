

NR="`head -n1 lp_state_nrs`"
if [ $? -eq 0 ]; then
    # gnuplot -e "set key autotitle columnhead; plot for [col=2:$((NR+1))] 'lp_v_gamma' using 1:col with lines; set key title 'LP V_{\gamma}(s) estimates'; pause mouse close; " &
    # gnuplot -e "set key autotitle columnhead; plot for [col=2:$((NR+1))] 'lp_v' using 1:col with lines; set key title 'LP V(s) estimates'; pause mouse close; " &
    gnuplot -e "set key autotitle columnhead; plot for [col=2:$((NR+1))] 'lp_e' using 1:col with lines; set key title 'LP Error Term estimates'; pause mouse close; " &
    gnuplot -e "set key autotitle columnhead; plot for [col=2:$((NR+1))] 'lp_gain_and_e' using 1:col with lines; set key title 'LP Gain and Error Term estimates'; pause mouse close; " &

    # gnuplot -e "set key autotitle columnhead; plot for [col=2:$((2*NR+1))] 'lp_v_gamma_delta' using 1:col with lines; set key title 'd/d{\gamma} of  LP V_{\gamma}(s) estimates'; pause mouse close; " &
    # gnuplot -e "set key autotitle columnhead; plot for [col=2:$((2*NR+1))] 'lp_v_delta' using 1:col with lines; set key title 'd/d{\gamma} of LP V(s) estimates'; pause mouse close; " &
    # gnuplot -e "set key autotitle columnhead; plot for [col=2:$((2*NR+1))] 'lp_e_delta' using 1:col with lines; set key title 'd/d{\gamma} of LP Error Term estimates'; pause mouse close; " &
    # gnuplot -e "set key autotitle columnhead; plot for [col=2:$((2*NR+1))] 'lp_gain_and_e_delta' using 1:col with lines; set key title 'd/d{\gamma} of LP Gain and Error Term estimates'; pause mouse close; " &

fi
