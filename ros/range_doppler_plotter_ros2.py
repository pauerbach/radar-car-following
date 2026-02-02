
import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt

from radar_msgs.msg import RadarFrame

from helpers.DopplerAlgo import *

class Draw:
    # Represents drawing for example
    #
    # Draw is done for each antenna, and each antenna is represented for
    # other subplot

    def __init__(self, max_speed_m_s, max_range_m, num_ant):
        # max_range_m:   maximum supported range
        # max_speed_m_s: maximum supported speed
        # num_ant:      number of available antennas
        self._h = []
        self._max_speed_m_s = max_speed_m_s
        self._max_range_m = max_range_m
        self._num_ant = num_ant

        plt.ion()

        self._fig, ax = plt.subplots(nrows=1, ncols=num_ant, figsize=((num_ant + 1) // 2, 2))
        # self._fig, ax = plt.subplots(nrows=1, ncols=num_ant, figsize=(8,6))
        if (num_ant == 1):
            self._ax = [ax]
        else:
            self._ax = ax

        self._fig.canvas.manager.set_window_title("Doppler")
        self._fig.set_size_inches(3 * num_ant + 1, 3 + 1 / num_ant)
        self._fig.canvas.mpl_connect('close_event', self.close)
        self._is_window_open = True

    def _draw_first_time(self, data_all_antennas):
        # First time draw
        #
        # It computes minimal, maximum value and draw data for all antennas
        # in same scale
        # data_all_antennas: array of raw data for each antenna

        minmin = min([np.min(data) for data in data_all_antennas])
        maxmax = max([np.max(data) for data in data_all_antennas])

        for i_ant in range(self._num_ant):
            data = data_all_antennas[i_ant]
            h = self._ax[i_ant].imshow(
                data,
                vmin=minmin, vmax=maxmax,
                #cmap='hot',
                extent=(-self._max_speed_m_s,
                        self._max_speed_m_s,
                        0,
                        self._max_range_m),
                aspect='auto',
                origin='lower')
            self._h.append(h)

            self._ax[i_ant].set_xlabel("velocity (m/s)")
            self._ax[i_ant].set_ylabel("distance (m)")
            self._ax[i_ant].set_title("antenna #" + str(i_ant))
        self._fig.subplots_adjust(right=0.8)
        cbar_ax = self._fig.add_axes([0.85, 0.0, 0.03, 1])

        cbar = self._fig.colorbar(self._h[0], cax=cbar_ax)
        cbar.ax.set_ylabel("magnitude (dB)")

    def _draw_next_time(self, data_all_antennas):
        # data_all_antennas: array of raw data for each antenna

        for i_ant in range(0, self._num_ant):
            data = data_all_antennas[i_ant]
            self._h[i_ant].set_data(data)

    def draw(self, data_all_antennas):
        # Draw plots for all antennas
        # data_all_antennas: array of raw data for each antenna
        if self._is_window_open:
            if len(self._h) == 0:  # handle the first run
                self._draw_first_time(data_all_antennas)
            else:
                self._draw_next_time(data_all_antennas)

            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()

    def close(self, event=None):
        if self.is_open():
            self._is_window_open = False
            plt.close(self._fig)
            plt.close('all')  # Needed for Matplotlib ver: 3.4.0 and 3.4.1
            print('Application closed!')

    def is_open(self):
        return self._is_window_open

def linear_to_dB(x):
    return 20 * np.log10(abs(x))

class RadarFrameSubscriber(Node):

    def __init__(self):
        super().__init__('radar_frame_subscriber')

        # objects to compute and visualize the range doppler map
        self.doppler = None
        self.draw = None

        self.subscription = self.create_subscription(
            RadarFrame,
            'radar_data',
            self.listener_callback,
            10
        )

        self.get_logger().info('RadarFrame subscriber node has started')

    def listener_callback(self, msg: RadarFrame):

        if self.doppler is None:
            self.doppler = DopplerAlgo(msg.num_samples, msg.num_chirps, msg.num_antenna, mti_alpha=1.0)
            self.draw = Draw(msg.max_doppler, msg.max_range, 1)

        # reshape data into original radar cube
        data = np.array(msg.data).reshape(msg.num_antenna, msg.num_chirps, msg.num_samples)

        data_all_antennas = []
        for i_ant in range(0, msg.num_antenna):  # for each antenna
            mat = data[i_ant, :, :]
            dfft_dbfs = linear_to_dB(self.doppler.compute_doppler_map(mat, i_ant))
            data_all_antennas.append(dfft_dbfs)

        data_full = np.array(data_all_antennas)
        data_full = np.average(data_full, 0)

        self.draw.draw([data_full])

def main(args=None):
    rclpy.init(args=args)

    node = RadarFrameSubscriber()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()