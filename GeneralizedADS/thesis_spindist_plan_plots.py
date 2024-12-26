#run from beavercube-rpi-software folder

from sat_ADCS_helpers import *
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
Xset_lqr = np.loadtxt('rw1_manchester_tests_prop/trial99/planned_state.csv', delimiter=',')
Uset_lqr = np.loadtxt('rw1_manchester_tests_prop/trial99/planned_command.csv', delimiter=',')
theta_set_pvg = np.loadtxt('rw1_manchester_tests_prop/trial99/theta_set_planned.csv', delimiter=',')

plt.figure()
ax = plt.subplot()
ax.plot(theta_set_pvg,label='plan v goal')
# ax.legend()
plt.title("Pointing Error")
plt.xlabel('Time (s)')
plt.ylabel("Error (deg)")
plt.draw()
plt.pause(1)
plt.savefig("plan_point.png")
plt.draw()
plt.pause(1)

lqr_times = np.array(range(501))

plot_the_thing((180.0/math.pi)*Xset_lqr[0:3,:],norm=True,title="Angular Velocity",ylabel="Angular Velocity (deg/s)",xlabel='Time (s)',xdata = np.array(lqr_times),save = False,plot_now = True, save_name = "tmp")

handles, labels = plt.gca().get_legend_handles_labels()
real_legend = plt.legend(handles=[handles[0], handles[1], handles[2],handles[3]], labels = [r"$\omega_x$",r"$\omega_y$",r"$\omega_z$",r"$|\boldsymbol{\omega}|$"],
    loc="lower right", bbox_to_anchor=(1, 0)
)
plt.gca().add_artist(real_legend)  # Add this legend to the plot
plt.draw()
plt.pause(1)
plt.savefig("plan_av.png")
plt.draw()
plt.pause(1)


plot_the_thing(Xset_lqr[3:7,:],norm=False,title="Quaternion",ylabel="Quaternion",xlabel='Time (s)',xdata = np.array(lqr_times),save = False,plot_now = True, save_name = "tmp")

handles, labels = plt.gca().get_legend_handles_labels()
real_legend = plt.legend(handles=[handles[0], handles[1], handles[2],handles[3]], labels = [r"$q_0$",r"$q_1$",r"$q_2$",r"$q_3$"],
    loc="lower right", bbox_to_anchor=(1, 0)
)
plt.gca().add_artist(real_legend)  # Add this legend to the plot
plt.draw()
plt.pause(1)
plt.savefig("plan_quat.png")
plt.draw()
plt.pause(1)

plot_the_thing(Uset_lqr[0:3,:],norm=True,title="MTQ Command",ylabel="MTQ Command (Am$^2$)",xlabel='Time (s)',xdata = np.array(lqr_times),save = False,plot_now = True, save_name = "tmp")

handles, labels = plt.gca().get_legend_handles_labels()
real_legend = plt.legend(handles=[handles[0], handles[1], handles[2],handles[3]], labels = [r"$m_x$",r"$m_y$",r"$m_z$",r"$|\mathbf{m}|$"],
    loc="lower right", bbox_to_anchor=(1, 0)
)
plt.gca().add_artist(real_legend)  # Add this legend to the plot
plt.draw()
plt.pause(1)
plt.savefig("plan_mtq.png")
plt.draw()
plt.pause(1)

plot_the_thing(Uset_lqr[3,:],norm=False,title="RW Command",ylabel="RW Command (Nm)",xlabel='Time (s)',xdata = np.array(lqr_times),save = False,plot_now = True, save_name = "tmp")
handles, labels = plt.gca().get_legend_handles_labels()
real_legend = plt.legend(handles=[handles[0]], labels = [r"$\tau_{\text{RW}}$"],
    loc="lower right", bbox_to_anchor=(1, 0)
)
plt.gca().add_artist(real_legend)  # Add this legend to the plot
plt.draw()
plt.pause(1)
plt.savefig("plan_rw.png")
plt.draw()
plt.pause(1)
plot_the_thing(Xset_lqr[7,:],title="RW-Stored Angular Momentum",ylabel="Stored Momentum (Nms)",xlabel='Time (s)',xdata = np.array(lqr_times),save = False,plot_now = True, save_name = "tmp")
handles, labels = plt.gca().get_legend_handles_labels()
real_legend = plt.legend(handles=[handles[0]], labels = [r"$h_{\text{RW}}$"],
    loc="lower right", bbox_to_anchor=(1, 0)
)
plt.gca().add_artist(real_legend)  # Add this legend to the plot
plt.draw()
plt.pause(1)
plt.savefig("plan_am.png")
plt.draw()
plt.pause(1)
