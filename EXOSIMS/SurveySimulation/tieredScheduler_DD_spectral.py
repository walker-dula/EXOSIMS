from EXOSIMS.SurveySimulation.tieredScheduler_DD import tieredScheduler_DD
import EXOSIMS, os
import astropy.units as u
import astropy.constants as const
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import time
import copy

class tieredScheduler_DD_spectral(tieredScheduler_DD):
    """tieredScheduler_DD_spectral
    
    This class implements a tiered scheduler that independantly schedules the observatory
    while the starshade slews to its next target.
    """

    def __init__(self, **specs):
        
        tieredScheduler_DD.__init__(self, **specs)


    def run_sim(self):
        """Performs the survey simulation 
        
        Returns:
            mission_end (string):
                Message printed at the end of a survey simulation.
        
        """
        
        OS = self.OpticalSystem
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping

        self.phase1_end = TK.missionStart + 365*u.d
        
        # TODO: start using this self.currentSep
        # set occulter separation if haveOcculter
        self.currentSep = Obs.occulterSep
        
        # Choose observing modes selected for detection (default marked with a flag),
        detMode = filter(lambda mode: mode['detectionMode'] == True, OS.observingModes)[0]
        # and for characterization (default is first spectro/IFS mode)
        spectroModes = filter(lambda mode: 'spec' in mode['inst']['name'], OS.observingModes)
        if np.any(spectroModes):
            charMode = spectroModes[0]
        # if no spectro mode, default char mode is first observing mode
        else:
            charMode = OS.observingModes[0]
        
        # Begin Survey, and loop until mission is finished
        self.logger.info('OB%s: survey beginning.'%(TK.OBnumber+1))
        print 'OB%s: survey beginning.'%(TK.OBnumber+1)
        t0 = time.time()
        sInd = None
        occ_sInd = None
        cnt = 0
        self.occ_arrives = TK.currentTimeAbs
        while not TK.mission_is_over():
             
            # Acquire the NEXT TARGET star index and create DRM
            prev_occ_sInd = occ_sInd
            DRM, sInd, occ_sInd, t_det, sd, occ_sInds = self.next_target(sInd, occ_sInd, detMode, charMode)
            assert t_det !=0, "Integration time can't be 0."

            if sInd is not None and (TK.currentTimeAbs + t_det) >= self.occ_arrives and np.any(occ_sInds):
                sInd = occ_sInd
                self.ready_to_update = True

            time2arrive = self.occ_arrives - TK.currentTimeAbs
            
            if sInd is not None:
                cnt += 1

                # clean up revisit list when one occurs to prevent repeats
                if np.any(self.starRevisit) and np.any(np.where(self.starRevisit[:,0] == float(sInd))):
                    s_revs = np.where(self.starRevisit[:,0] == float(sInd))[0]
                    dt_max = 1.*u.week
                    t_revs = np.where(self.starRevisit[:,1]*u.day - TK.currentTimeNorm < dt_max)[0]
                    self.starRevisit = np.delete(self.starRevisit, np.intersect1d(s_revs,t_revs),0)

                # get the index of the selected target for the extended list
                if TK.currentTimeNorm > TK.missionLife and self.starExtended.shape[0] == 0:
                    for i in range(len(self.DRM)):
                        if np.any([x == 1 for x in self.DRM[i]['plan_detected']]):
                            self.starExtended = np.hstack((self.starExtended, self.DRM[i]['star_ind']))
                            self.starExtended = np.unique(self.starExtended)
                
                # Beginning of observation, start to populate DRM
                DRM['OB#'] = TK.OBnumber+1
                DRM['Obs#'] = cnt
                DRM['star_ind'] = sInd
                DRM['arrival_time'] = TK.currentTimeNorm.to('day').value
                pInds = np.where(SU.plan2star == sInd)[0]
                DRM['plan_inds'] = pInds.astype(int).tolist()

                if sInd == occ_sInd:
                    # wait until expected arrival time is observed
                    if time2arrive > 0*u.d:
                        TK.allocate_time(time2arrive.to('day'))
                        if time2arrive > 1*u.d:
                            self.GAtime = self.GAtime + time2arrive.to('day')

                TK.obsStart = TK.currentTimeNorm.to('day')

                self.logger.info('  Observation #%s, target #%s/%s with %s planet(s), mission time: %s'\
                        %(cnt, sInd+1, TL.nStars, len(pInds), TK.obsStart.round(2)))
                print '  Observation #%s, target #%s/%s with %s planet(s), mission time: %s'\
                        %(cnt, sInd+1, TL.nStars, len(pInds), TK.obsStart.round(2))
                
                if sInd != occ_sInd:
                    # PERFORM DETECTION and populate revisit list attribute.
                    # First store fEZ, dMag, WA
                    if np.any(pInds):
                        DRM['det_fEZ'] = SU.fEZ[pInds].to('1/arcsec2').value.tolist()
                        DRM['det_dMag'] = SU.dMag[pInds].tolist()
                        DRM['det_WA'] = SU.WA[pInds].to('mas').value.tolist()
                    detected, det_fZ, det_systemParams, det_SNR, FA = self.observation_detection(sInd, t_det, detMode)
                    if np.any(detected):
                        print '  Det. results are: %s'%(detected)
                    # update GAtime
                    self.GAtime = self.GAtime + t_det.to('day')*.07
                    # populate the DRM with detection results
                    DRM['det_time'] = t_det.to('day')
                    DRM['det_status'] = detected
                    DRM['det_SNR'] = det_SNR
                    DRM['det_fZ'] = det_fZ.to('1/arcsec2')
                    DRM['det_params'] = det_systemParams
                    DRM['FA_det_status'] = int(FA)
                
                elif sInd == occ_sInd:
                    # PERFORM CHARACTERIZATION and populate spectra list attribute.
                    # First store fEZ, dMag, WA, and characterization mode
                    occ_pInds = np.where(SU.plan2star == occ_sInd)[0]
                    sInd = occ_sInd

                    DRM['slew_time'] = self.occ_slewTime.to('day').value
                    DRM['slew_angle'] = self.occ_sd.to('deg').value
                    slew_mass_used = self.occ_slewTime*Obs.defburnPortion*Obs.flowRate
                    DRM['slew_dV'] = (self.occ_slewTime*self.ao*Obs.defburnPortion).to('m/s').value
                    DRM['slew_mass_used'] = slew_mass_used.to('kg')
                    Obs.scMass = Obs.scMass - slew_mass_used
                    DRM['scMass'] = Obs.scMass.to('kg')

                    DRM['char_info'] = []

                    self.logger.info('  Starshade and telescope aligned at target star')
                    print '  Starshade and telescope aligned at target star'

                     # PERFORM CHARACTERIZATION and populate spectra list attribute
                    characterized, char_fZ, char_systemParams, char_SNR, char_intTime, minimodes = \
                            self.observation_characterization(sInd, charMode)

                    if np.any(characterized):
                        print '  Char. results are: %s'%(characterized)
                    assert char_intTime != 0, "Integration time can't be 0."

                    for j, mode in enumerate(minimodes):
                        char_data = {}
                        if np.any(occ_pInds):
                            char_data['char_fEZ'] = SU.fEZ[occ_pInds].to('1/arcsec2').value.tolist()
                            char_data['char_dMag'] = SU.dMag[occ_pInds].tolist()
                            char_data['char_WA'] = SU.WA[occ_pInds].to('mas').value.tolist()
                        char_data['char_mode'] = dict(mode)
                        del char_data['char_mode']['inst'], char_data['char_mode']['syst']

                        # update the occulter wet mass
                        if OS.haveOcculter == True and char_intTime is not None:
                            DRM = self.update_occulter_mass(DRM, sInd, char_intTime, 'char')
                        FA = False
                        # populate the DRM with characterization results
                        char_data['char_time'] = char_intTime.to('day') if char_intTime else 0.*u.day
                        #DRM['char_counts'] = self.sInd_charcounts[sInd]
                        char_data['char_status'] = characterized[:,j]
                        char_data['char_SNR'] = char_SNR[:,j]
                        char_data['char_fZ'] = char_fZ.to('1/arcsec2')
                        char_data['char_params'] = char_systemParams
                        # populate the DRM with FA results
                        char_data['FA_char_status'] = characterized[-1] if FA else 0
                        char_data['FA_char_SNR'] = char_SNR[-1] if FA else 0.
                        char_data['FA_char_fEZ'] = self.lastDetected[sInd,1][-1]/u.arcsec**2 if FA else 0./u.arcsec**2
                        char_data['FA_char_dMag'] = self.lastDetected[sInd,2][-1] if FA else 0.
                        char_data['FA_char_WA'] = self.lastDetected[sInd,3][-1]*u.arcsec if FA else 0.*u.arcsec
                        DRM['char_info'].append(char_data)
                    DRM['FA_det_status'] = int(FA)

                    # add star back into the revisit list
                    if np.any(characterized):
                        char = np.where(characterized)[0]
                        pInds = np.where(SU.plan2star == sInd)[0]
                        smin = np.min(SU.s[pInds[char]])
                        pInd_smin = pInds[np.argmin(SU.s[pInds[char]])]

                        Ms = TL.MsTrue[sInd]
                        sp = smin
                        Mp = SU.Mp[pInd_smin]
                        mu = const.G*(Mp + Ms)
                        T = 2.*np.pi*np.sqrt(sp**3/mu)
                        t_rev = TK.currentTimeNorm + T/2.
                        revisit = np.array([sInd, t_rev.to('day').value])
                        if self.starRevisit.size == 0:
                            self.starRevisit = np.array([revisit])
                        else:
                            self.starRevisit = np.vstack((self.starRevisit, revisit))

                self.goal_GAtime = self.GA_percentage * TK.currentTimeNorm.to('day')
                goal_GAdiff = self.goal_GAtime - self.GAtime

                # allocate extra time to GA if we are falling behind
                if goal_GAdiff > 1*u.d:
                    print 'Allocating time %s to general astrophysics'%(goal_GAdiff)
                    self.GAtime = self.GAtime + goal_GAdiff
                    TK.allocate_time(goal_GAdiff)

                # Append result values to self.DRM
                self.DRM.append(DRM)

                # Calculate observation end time
                TK.obsEnd = TK.currentTimeNorm.to('day')

                # With prototype TimeKeeping, if no OB duration was specified, advance
                # to the next OB with timestep equivalent to time spent on one target
                if np.isinf(TK.OBduration):
                    obsLength = (TK.obsEnd-TK.obsStart).to('day')
                    TK.next_observing_block(dt=obsLength)
                
                # With occulter, if spacecraft fuel is depleted, exit loop
                if Obs.scMass < Obs.dryMass:
                    print 'Total fuel mass exceeded at %s' %TK.obsEnd.round(2)
                    break

        else:
            dtsim = (time.time()-t0)*u.s
            mission_end = "Mission complete: no more time available.\n"\
                    + "Simulation duration: %s.\n" %dtsim.astype('int')\
                    + "Results stored in SurveySimulation.DRM (Design Reference Mission)."

            self.logger.info(mission_end)
            print mission_end

            return mission_end


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

        # get the last detected planets, and check if there was a FA
        det = np.ones(pInds.size, dtype=bool)
        fEZs = SU.fEZ[pInds].to('1/arcsec2').value
        dMags = SU.dMag[pInds]
        WAs = SU.WA[pInds].to('arcsec').value
        minimodes = []

        # then, calculate number of bins
        lam0 = mode['lam']
        R = mode['inst']['Rs']       # spectral resolution
        deltaLam = lam0/R
        BW0_min = lam0 * (1 - mode['BW']/2)
        BW0_max = lam0 * (1 + mode['BW']/2)
        lam = BW0_min + deltaLam/2
        num_SNR_bins = int((BW0_max - BW0_min)/deltaLam)

        FA = (det.size == pInds.size + 1)
        if FA == True:
            pIndsDet = np.append(pInds, -1)[det]
        else:
            pIndsDet = pInds[det]

        # initialize outputs, and check if any planet to characterize
        characterized = np.zeros(det.size, dtype=int)
        fZ = 0./u.arcsec**2
        systemParams = SU.dump_system_params(sInd) # write current system params by default
        SNR = np.zeros((len(det), num_SNR_bins))
        intTime = None
        if len(det) == 0: # nothing to characterize
            if sInd not in self.sInd_charcounts.keys():
                self.sInd_charcounts[sInd] = characterized
            return characterized, fZ, systemParams, SNR, intTime
        
        # look for last detected planets that have not been fully characterized
        if (FA == False): # only true planets, no FA
            tochar = (self.fullSpectra[pIndsDet] != -2)
        else: # mix of planets and a FA
            truePlans = pIndsDet[:-1]
            tochar = np.append((self.fullSpectra[truePlans] == 0), True)
        
        # 1/ find spacecraft orbital START position and check keepout angle
        if np.any(tochar):
            # start times
            startTime = TK.currentTimeAbs
            startTimeNorm = TK.currentTimeNorm
            # planets to characterize
            tochar[tochar] = Obs.keepout(TL, sInd, startTime, mode)

        # 2/ if any planet to characterize, find the characterization times
        if np.any(tochar):
            # propagate the whole system to match up with current time
            # calculate characterization times at the detected fEZ, dMag, and WA
            fZ = ZL.fZ(Obs, TL, sInd, startTime, mode)
            # fEZ = self.lastDetected[sInd,1][tochar]/u.arcsec**2
            # dMag = self.lastDetected[sInd,2][tochar]
            # WA = self.lastDetected[sInd,3][tochar]*u.mas
            fEZ = fEZs[tochar]/u.arcsec**2
            dMag = dMags[tochar]
            WAp = WAs[tochar]*u.arcsec

            intTimes = np.zeros(len(pInds))*u.d
            # t_chars[tochar] = OS.calc_intTime(TL, sInd, fZ, fEZ, dMag, WA, mode)
            for i,j in enumerate(WAp):
                if tochar[i]:
                    intTimes[i] = self.calc_int_inflection([sInd], fEZ[i], startTime, j, mode, ischar=True)[0]

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
            # SNRfa = []
            # if pIndsChar[-1] == -1:
            #     fEZ = fEZs[-1]/u.arcsec**2
            #     dMag = dMags[-1]
            #     WA = WAs[-1]*u.arcsec
            #     C_p, C_b, C_sp = OS.Cp_Cb_Csp(TL, sInd, fZ, fEZ, dMag, WA, mode)
            #     S = (C_p*intTime).decompose().value
            #     N = np.sqrt((C_b*intTime + (C_sp*intTime)**2).decompose().value)
            #     SNRfa = S/N if N > 0 else 0.
            
            # save all SNRs (planets and FA) to one array
            SNRinds = np.where(det)[0][tochar]
            SNR[SNRinds,:] = SNRplans
            
            # now, store characterization status: 1 for full spectrum, 
            # -1 for partial spectrum, 0 for not characterized
            char = (SNR >= mode['SNR'])

            # initialize with full spectra
            characterized = char.astype(int)
            WAs = np.array([WAs,]*num_SNR_bins).transpose()
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
            charplans = characterized
            #self.fullSpectra[pInds[charplans == 1]] += 1
            #self.partialSpectra[pInds[charplans == -1]] += 1

        # in both cases (detection or false alarm), schedule a revisit 
        # based on minimum separation
        smin = np.min(SU.s[pInds[det]])
        Ms = TL.MsTrue[sInd]
        if smin is not None:
            sp = smin
            if np.any(det):
                pInd_smin = pInds[det][np.argmin(SU.s[pInds[det]])]
                Mp = SU.Mp[pInd_smin]
            else:
                Mp = SU.Mp.mean()
            mu = const.G*(Mp + Ms)
            T = 2.*np.pi*np.sqrt(sp**3/mu)
            t_rev = TK.currentTimeNorm + T/2.
        # otherwise, revisit based on average of population semi-major axis and mass
        else:
            sp = SU.s.mean()
            Mp = SU.Mp.mean()
            mu = const.G*(Mp + Ms)
            T = 2.*np.pi*np.sqrt(sp**3/mu)
            t_rev = TK.currentTimeNorm + 0.75*T

        # finally, populate the revisit list (NOTE: sInd becomes a float)
        revisit = np.array([sInd, t_rev.to('day').value])
        if self.starRevisit.size == 0:
            self.starRevisit = np.array([revisit])
        else:
            revInd = np.where(self.starRevisit[:,0] == sInd)[0]
            if revInd.size == 0:
                self.starRevisit = np.vstack((self.starRevisit, revisit))
            else:
                self.starRevisit[revInd,1] = revisit[1]

        return characterized.astype(int), fZ, systemParams, SNR, intTime, minimodes

