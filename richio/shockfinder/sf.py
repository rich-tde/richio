#  Copyright 2025 The RICHIO Contributors
#
#  This file is part of RICHIO.
#
#  RICHIO is free software: you can redistribute it and/or modify it under
#  the terms of the European Union Public License version 1.2 or later, as
#  published by the European Commission.
#
#  RICHIO is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
#  A PARTICULAR PURPOSE. See the European Union Public License for more details.
#
#  You should have received a copy of the EUPL in an/all official language(s) of
#  the European Union along with RICHIO.  If not, see <https://eupl.eu>.

#  Copyright 2025 The RICHIO Contributors
#
#  This file is part of RICHIO.
#
#  RICHIO is free software: you can redistribute it and/or modify it under
#  the terms of the European Union Public License version 1.2 or later, as
#  published by the European Commission.
#
#  RICHIO is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
#  A PARTICULAR PURPOSE. See the European Union Public License for more details.
#
#  You should have received a copy of the EUPL in an/all official language(s) of
#  the European Union along with RICHIO.  If not, see <https://eupl.eu>.
#
#  The shockfinder algorithm in RICHIO is based on Schaal+14
#  <https://arxiv.org/abs/1407.4117>.

import numpy as np


# ---------------------------------------------------------------------------- #
#                             Analytical Functions                             #
# ---------------------------------------------------------------------------- #

def delta(M, gamma=5/3):
    R = 1/((gamma - 1)/(gamma + 1) + 2/(gamma + 1)/M**2) # R = rho2/rho1
    delta = 2/(gamma*(gamma - 1) * M**2 * R) * ((2*gamma*M**2 - (gamma - 1))/(gamma + 1) - R**gamma)
    return delta

def R2M(R, gamma=5/3):
    """
    Mach number from compression ratio rho2/rho1, using the RH condition.
    """
    return 1/np.sqrt((gamma + 1)/(2*R) - (gamma - 1)/2)
# Alias
rho2rho1M = R2M

def M2R(M, gamma=5/3):
    """
    Compression ratio rho2/rho1 from Mach number, using the RH condition.
    """
    rho2rho1 = (gamma + 1)*M**2 / ((gamma - 1)*M**2 + 2)
    return rho2rho1
# Alias
Mrho2rho1 = M2R

def MT2T1(M, gamma=5/3):
    """
    Temperature jump ratio T2/T1 from Mach number, using the RH condition.
    """
    T2T1 = (2*gamma*M**2 - (gamma - 1)) * ((gamma - 1)*M**2 + 2) / ((gamma + 1)**2 * M**2)
    return T2T1

def MP2P1(M, gamma=5/3):
    """
    Pressure jump ratio P2/P1 from Mach number, using the RH condition.
    """
    P2P1 = (2*gamma*M**2)/(gamma + 1) - (gamma - 1)/(gamma + 1)
    return P2P1

def T2T1M(T2_T1, gamma):
    """ Find mach number from the temperature jump (T2_T1). """
    a = 2 * gamma * (gamma - 1)
    minusb = gamma * 2 - 6 * gamma + T2_T1 * (gamma + 1)**2 + 1
    M2 = (minusb + np.sqrt(minusb**2 + 8 * a * (gamma - 1))) / (2 * a)
    return np.sqrt(M2)

def P2P1M(P2_P1, gamma):
    """ Find mach number from the pressure jump (P2_P1). """
    M2 = (P2_P1 * (gamma + 1) + gamma - 1) / (2 * gamma)
    return np.sqrt(M2)

# ---------------------------------------------------------------------------- #
#                                  Shock Zone                                  #
# ---------------------------------------------------------------------------- #



# ---------------------------------------------------------------------------- #
#                                 Shock Surface                                #
# ---------------------------------------------------------------------------- #

