# ===========================================================================
# Copyright (C) 2022 Infineon Technologies AG
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ===========================================================================

import numpy as np


class CaponBeamforming:
    def __init__(self, num_antennas: int, num_beams: int = 27, max_angle_degrees: float = 45, d_by_lambda: float = 0.5):
        """Create a Digital Beam Forming object

        Parameters:
            - num_antennas:         number of (virtual) RX antennas
            - num_beams:            number of beams
            - max_angle_degrees:    maximum angle in degrees, angles will range
                                    from -max_angle_degrees .. +max_angle_degrees
            - d_by_lambda:          separation of RX antennas divided by the wavelength
        """

        angles_capon = np.linspace(-90, 90, num_beams)
        angles_rad = np.deg2rad(angles_capon)
        self.steering_vectors = np.exp(
            1j * 2 * np.pi * d_by_lambda * np.outer(np.arange(num_antennas), np.sin(angles_rad))
        )

        self.num_beams = num_beams

    def capon_beamforming(self, X):
        Rxx = np.dot(X, X.conj().T) / X.shape[1]  # covariance matrix
        Rxx_inv = np.linalg.pinv(Rxx)
        P_capon = np.zeros(self.steering_vectors.shape[1])
        for k in range(self.steering_vectors.shape[1]):
            a = self.steering_vectors[:, k][:, None]
            P_capon[k] = 1 / np.real(np.dot(a.conj().T, np.dot(Rxx_inv, a)))
        return P_capon

    def run(self, range_fft):
        num_samples, num_chirps, num_antennas = range_fft.shape

        range_angle_capon = np.zeros((num_samples, self.num_beams))
        for r in range(num_samples):
            # Extract Doppler snapshots from range FFT
            X = range_fft[r, :, :].T  # N_rx x N_c
            P = self.capon_beamforming(X)
            range_angle_capon[r, :] = P

        return range_angle_capon
