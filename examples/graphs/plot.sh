
FONTSCALE=1
INIT_GNUPLOT="set terminal wxt lw $FONTSCALE; set terminal wxt fontscale $FONTSCALE"

# Exp Smoothed reward
read -r -d '' VAR << EOM
for [col=3:3] 'season_forecast_low_aral' using 0:col with lines title 'ARAL Exp Smth Rew',
for [col=2:2] 'season_forecast_low_aral' using 0:col with lines title 'ARAL Learned Avg. Reward',
for [col=3:3] 'season_forecast_low_ql'   using 0:col with lines title 'QL Exp Smth Rew',
for [col=3:3] 'season_forecast_low_rl'   using 0:col with lines title 'RL Exp Smth Rew',
for [col=2:2] 'season_forecast_low_rl'   using 0:col with lines title 'RL Learned Avg. Reward'
EOM

START=2
END=3
gnuplot -e "$INIT_GNUPLOT; set title 'Demand Forecast Low Variability'; set key autotitle columnhead; plot $VAR; pause mouse close; " &


# Exp Smoothed reward
read -r -d '' VAR << EOM
for [col=3:3] 'season_forecast_medium_aral' using 0:col with lines title 'ARAL Exp Smth Rew',
for [col=2:2] 'season_forecast_medium_aral' using 0:col with lines title 'ARAL Learned Avg. Reward',
for [col=3:3] 'season_forecast_medium_ql'   using 0:col with lines title 'QL Exp Smth Rew',
for [col=3:3] 'season_forecast_medium_rl'   using 0:col with lines title 'RL Exp Smth Rew',
for [col=2:2] 'season_forecast_medium_rl'   using 0:col with lines title 'RL Learned Avg. Reward'
EOM

START=2
END=3
gnuplot -e "$INIT_GNUPLOT; set title 'Demand Forecast Medium Variability'; set key autotitle columnhead; plot $VAR; pause mouse close; " &


# Exp Smoothed reward
read -r -d '' VAR << EOM
for [col=3:3] 'season_forecast_high_aral' using 0:col with lines title 'ARAL Exp Smth Rew',
for [col=2:2] 'season_forecast_high_aral' using 0:col with lines title 'ARAL Learned Avg. Reward',
for [col=3:3] 'season_forecast_high_ql'   using 0:col with lines title 'QL Exp Smth Rew',
for [col=3:3] 'season_forecast_high_rl'   using 0:col with lines title 'RL Exp Smth Rew',
for [col=2:2] 'season_forecast_high_rl'   using 0:col with lines title 'RL Learned Avg. Reward'
EOM

START=2
END=3
gnuplot -e "$INIT_GNUPLOT; set title 'Demand Forecast High Variability'; set key autotitle columnhead; plot $VAR; pause mouse close; " &


# Exp Smoothed reward
read -r -d '' VAR << EOM
for [col=3:3] 'step_forecast_high_aral' using 0:col with lines title 'ARAL Exp Smth Rew',
for [col=2:2] 'step_forecast_high_aral' using 0:col with lines title 'ARAL Learned Avg. Reward',
for [col=3:3] 'step_forecast_high_ql'   using 0:col with lines title 'QL Exp Smth Rew',
for [col=3:3] 'step_forecast_high_rl'   using 0:col with lines title 'RL Exp Smth Rew',
for [col=2:2] 'step_forecast_high_rl'   using 0:col with lines title 'RL Learned Avg. Reward'
EOM

START=2
END=3
gnuplot -e "$INIT_GNUPLOT; set title 'Step Forecast High Variability'; set key autotitle columnhead; plot $VAR; pause mouse close; " &
