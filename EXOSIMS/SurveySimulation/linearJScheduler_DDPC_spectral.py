from EXOSIMS.SurveySimulation.linearJScheduler_DDPC import linearJScheduler_DDPC
import astropy.units as u
import numpy as np
import itertools

class linearJScheduler_DDPC_spectral(linearJScheduler_DDPC):
    """linearJScheduler 2
    
    This class implements the linear cost function scheduler described
    in Savransky et al. (2010).
    
        Args:
        coeffs (iterable 3x1):
            Cost function coefficients: slew distance, completeness, target list coverage
        
        \*\*specs:
            user specified values
    
    """

    def __init__(self, **specs):
        
        linearJScheduler_DDPC.__init__(self, **specs)


    def observation_characterization(self, sInd, mode):
        """Finds if characterizations are possible and relevant information
        
        Args:
            sInd (integer):
                Integer index of the star of interest
            mode (dict):
                Selected observing mode for characterization
        
        Returns:
            characterized (integer list):
                Characterization status for each planet orbiting the observed 
                target star including False Alarm if any, where 1 is full spectrum, 
                -1 partial spectrum, and 0 not characterized
            fZ (astropy Quantity):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            systemParams (dict):
                Dictionary of time-dependant planet properties averaged over the 
                duration of the integration
            SNR (float ndarray):
                Characterization signal-to-noise ratio of the observable planets. 
                Defaults to None.
            intTime (astropy Quantity):
                Selected star characterization time in units of day. Defaults to None.
        
        """
        
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping
        
        # find indices of planets around the target
        pInds = np.where(SU.plan2star == sInd)[0]
        
        # get the detected status, and check if there was a FA
        det = self.lastDetected[sInd,0]
        minimodes = []

        # then, calculate number of bins
        lam0 = mode['lam']
        R = mode['inst']['Rs']       # spectral resolution
        deltaLam = lam0/R
        BW0_min = lam0 * (1 - mode['BW']/2)
        BW0_max = lam0 * (1 + mode['BW']/2)
        lam = BW0_min + deltaLam/2
        num_SNR_bins = int((BW0_max - BW0_min)/deltaLam)

        FA = (len(det) == len(pInds) + 1)
        if FA == True:
            pIndsDet = np.append(pInds, -1)[det]
        else:
            pIndsDet = pInds[det]
        
        # initialize outputs, and check if there's anything (planet or FA) to characterize
        characterized = np.zeros(len(det), dtype=int)
        fZ = 0./u.arcsec**2
        systemParams = SU.dump_system_params(sInd) # write current system params by default
        SNR = np.zeros((len(det), num_SNR_bins))
        intTime = None
        if len(det) == 0: # nothing to characterize
            return characterized, fZ, systemParams, SNR, intTime
        
        # look for last detected planets that have not been fully characterized
        if (FA == False): # only true planets, no FA
            tochar = (self.fullSpectra[pIndsDet] != -2)
        else: # mix of planets and a FA
            truePlans = pIndsDet[:-1]
            tochar = np.append((self.fullSpectra[truePlans] == 0), True)
        print(tochar)
        
        # 1/ find spacecraft orbital START position and check keepout angle
        if np.any(tochar):
            # start times
            startTime = TK.currentTimeAbs
            startTimeNorm = TK.currentTimeNorm
            # planets to characterize
            tochar[tochar] = Obs.keepout(TL, sInd, startTime, mode)
        print(tochar)

        # 2/ if any planet to characterize, find the characterization times
        if np.any(tochar):
            # propagate the whole system to match up with current time
            # calculate characterization times at the detected fEZ, dMag, and WA
            fZ = ZL.fZ(Obs, TL, sInd, startTime, mode)
            fEZ = self.lastDetected[sInd,1][tochar]/u.arcsec**2
            dMag = self.lastDetected[sInd,2][tochar]
            WA = self.lastDetected[sInd,3][tochar]*u.mas

            intTimes = np.zeros(len(pInds))*u.d
            # t_chars[tochar] = OS.calc_intTime(TL, sInd, fZ, fEZ, dMag, WA, mode)
            intTimes = np.zeros(len(tochar))*u.day
            intTimes[tochar] = OS.calc_intTime(TL, sInd, fZ, fEZ, dMag, WA, mode)
            print(intTimes)

            # add a predetermined margin to the integration times
            intTimes = intTimes*(1 + self.charMargin)
            # apply time multiplier
            totTimes = intTimes*(mode['timeMultiplier'])
            # end times
            endTimes = startTime + totTimes
            endTimesNorm = startTimeNorm + totTimes
            # planets to characterize
            tochar = ((totTimes > 0) & (totTimes <= OS.intCutoff) & 
                    (endTimesNorm <= TK.OBendTimes[TK.OBnumber]))
        
        # 3/ is target still observable at the end of any char time?
        if np.any(tochar) and Obs.checkKeepoutEnd:
            tochar[tochar] = Obs.keepout(TL, sInd, endTimes[tochar], mode)
        
        # 4/ if yes, perform the characterization for the maximum char time
        if np.any(tochar):
            intTime = np.max(intTimes[tochar])
            pIndsChar = pIndsDet[tochar]
            log_char = '   - Charact. planet(s) %s (%s/%s detected)'%(pIndsChar, 
                    len(pIndsChar), len(pIndsDet))
            self.logger.info(log_char)
            print log_char
            
            # SNR CALCULATION:
            # first, calculate SNR for observable planets (without false alarm)
            planinds = pIndsChar[:-1] if pIndsChar[-1] == -1 else pIndsChar

            SNRplans = np.zeros((len(planinds), num_SNR_bins))

            # create a new submode for each spectral bin
            for j in  range(num_SNR_bins):
                minimode = copy.deepcopy(mode)
                BW = (1 - ((lam - deltaLam/2)/lam)) * 2
                minimode['lam'] = lam
                minimode['BW'] = BW
                minimodes.append(minimode)

            if len(planinds) > 0:
                # initialize arrays for SNR integration
                fZs = np.zeros((self.ntFlux, num_SNR_bins))/u.arcsec**2
                systemParamss = np.empty(self.ntFlux, dtype='object')
                Ss = np.zeros((self.ntFlux, len(planinds), num_SNR_bins))
                Ns = np.zeros((self.ntFlux, len(planinds), num_SNR_bins))
                # integrate the signal (planet flux) and noise
                dt = intTime/self.ntFlux
                for i in range(self.ntFlux):
                    TK.allocate_time(dt/2.)
                    for j, minimode in enumerate(minimodes):
                        # allocate first half of dt
                        # calculate current zodiacal light brightness
                        fZs[i,j] = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs, minimode)[0]
                        # propagate the system to match up with current time
                        SU.propag_system(sInd, TK.currentTimeNorm - self.propagTimes[sInd])
                        self.propagTimes[sInd] = TK.currentTimeNorm
                        # save planet parameters
                        systemParamss[i] = SU.dump_system_params(sInd)
                        # calculate signal and noise (electron count rates)
                        Ss[i,:,j], Ns[i,:,j] = self.calc_signal_noise(sInd, planinds, dt, minimode, 
                                fZ=fZs[i,j])
                        # allocate second half of dt
                        lam += deltaLam
                    TK.allocate_time(dt/2.)
                
                # average output parameters
                fZ = np.mean(fZs)
                systemParams = {key: sum([systemParamss[x][key]
                        for x in range(self.ntFlux)])/float(self.ntFlux)
                        for key in sorted(systemParamss[0])}
                # calculate planets SNR
                S = Ss.sum(0)
                N = Ns.sum(0)
                SNRplans[N > 0] = S[N > 0]/N[N > 0]
                print(SNRplans)
                # allocate extra time for timeMultiplier
                extraTime = intTime*(mode['timeMultiplier'] - 1)
                TK.allocate_time(extraTime)
            
            # if only a FA, just save zodiacal brightness in the middle of the integration
            else:
                totTime = intTime*(mode['timeMultiplier'])
                TK.allocate_time(totTime/2.)
                fZ = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs, mode)[0]
                TK.allocate_time(totTime/2.)
            
            # calculate the false alarm SNR (if any)
            SNRfa = []
            if pIndsChar[-1] == -1:
                fEZ = self.lastDetected[sInd,1][-1]/u.arcsec**2
                dMag = self.lastDetected[sInd,2][-1]
                WA = self.lastDetected[sInd,3][-1]*u.arcsec
                for minimode in minimodes:
                    C_p, C_b, C_sp = OS.Cp_Cb_Csp(TL, sInd, fZ, fEZ, dMag, WA, minimode)
                    S = (C_p*intTime).decompose().value
                    N = np.sqrt((C_b*intTime + (C_sp*intTime)**2).decompose().value)
                    snr_fa = S/N if N > 0 else 0.
                    SNRfa.append(snr_fa)
            
            # save all SNRs (planets and FA) to one array
            SNRinds = np.where(det)[0][tochar]
            if pIndsChar[-1] == -1:
                SNR[SNRinds] = np.concatinate((SNRplans, SNRfa), axis=1)
            else:
                SNR[SNRinds,:] = SNRplans
            
            # now, store characterization status: 1 for full spectrum, 
            # -1 for partial spectrum, 0 for not characterized
            char = (SNR >= mode['SNR'])

            # initialize with full spectra
            characterized = char.astype(int)
            # find the current WAs of characterized planets
            WAs = np.array([WAs,]*num_SNR_bins).transpose()
            if FA:
                WAs = np.concatinate((WAs, WAs[-1,:]*u.arcsec), axis=1)
            # check for partial spectra
            for j, minimode in enumerate(minimodes):
                WAchar = WAs[:,j][char[:,j]]*u.arcsec
                IWA_max = minimode['IWA']*(1 + minimode['BW']/2.)
                OWA_min = minimode['OWA']*(1 - minimode['BW']/2.)
                char[:,j][char[:,j]] = (WAchar < IWA_max) | (WAchar > OWA_min)
            characterized[char] = -1
            all_full = np.copy(characterized)
            all_full[char] = 0
            if sInd not in self.sInd_charcounts.keys():
                self.sInd_charcounts[sInd] = all_full
            else:
                self.sInd_charcounts[sInd] = self.sInd_charcounts[sInd] + all_full
            # encode results in spectra lists (only for planets, not FA)
            charplans = characterized[:-1,:] if FA else characterized
            self.fullSpectra[pInds[charplans == 1]] += 1
            self.partialSpectra[pInds[charplans == -1]] += 1

        print(characterized.astype(int))
        return characterized.astype(int), fZ, systemParams, SNR, intTime
