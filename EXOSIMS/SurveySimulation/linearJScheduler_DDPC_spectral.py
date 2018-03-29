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
        detModes = filter(lambda mode: 'imag' in mode['inst']['name'], OS.observingModes)
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
            DRM, sInd, occ_sInd, t_det, sd, occ_sInds, dmode = self.next_target(sInd, occ_sInd, detModes, charMode)
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
                    detected, det_fZ, det_systemParams, det_SNR, FA = self.observation_detection(sInd, t_det, dmode)
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
                    DRM['det_mode'] = dict(dmode)
                    DRM['FA_det_status'] = int(FA)
                    del DRM['det_mode']['inst'], DRM['det_mode']['syst']
                
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
                        print '  Char. results are: %s'%(characterized.T)
                    assert char_intTime != 0, "Integration time can't be 0."

                    # update the occulter wet mass
                    if OS.haveOcculter == True and char_intTime is not None:
                        DRM = self.update_occulter_mass(DRM, sInd, char_intTime, 'char')

                    for j, mode in enumerate(minimodes):
                        char_data = {}
                        if np.any(occ_pInds):
                            char_data['char_fEZ'] = SU.fEZ[occ_pInds].to('1/arcsec2').value.tolist()
                            char_data['char_dMag'] = SU.dMag[occ_pInds].tolist()
                            char_data['char_WA'] = SU.WA[occ_pInds].to('mas').value.tolist()
                        char_data['char_mode'] = dict(mode)
                        del char_data['char_mode']['inst'], char_data['char_mode']['syst']

                        FA = False
                        # populate the DRM with characterization results
                        char_data['char_time'] = char_intTime.to('day') if char_intTime else 0.*u.day
                        #DRM['char_counts'] = self.sInd_charcounts[sInd]
                        char_data['char_status'] = characterized[:,j]
                        char_data['char_SNR'] = char_SNR[:,j]
                        char_data['char_fZ'] = char_fZ.to('1/arcsec2')
                        char_data['char_params'] = char_systemParams
                        DRM['char_info'].append(char_data)
                    DRM['FA_det_status'] = int(FA)

                    # add star back into the revisit list
                    if np.any(characterized):
                        char = np.unique(np.where(characterized)[0])
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


    def observation_characterization(self, sInd, modes):
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

        nmodes = len(modes)
        
        # find indices of planets around the target
        pInds = np.where(SU.plan2star == sInd)[0]
        
        # get the detected status, and check if there was a FA
        det = self.lastDetected[sInd,0]

        # then, calculate number of bins
        lam0_0 = modes[0]['lam']
        R = modes[0]['inst']['Rs']       # spectral resolution
        deltaLam = lam0_0/R
        BW0_min = lam0_0 * (1 - modes[0]['BW']/2)
        BW0_max = lam0_0 * (1 + modes[0]['BW']/2)
        lam1 = BW0_min + deltaLam/2
        num_SNR_bins1 = int((BW0_max - BW0_min)/deltaLam)

        lam0_1 = modes[1]['lam']
        R = modes[1]['inst']['Rs']       # spectral resolution
        deltaLam2 = lam0_1/R
        BW1_min = lam0_1 * (1 - modes[1]['BW']/2)
        BW1_max = lam0_1 * (1 + modes[1]['BW']/2)
        lam2 = BW1_min + deltaLam2/2
        num_SNR_bins2 = int((BW0_max - BW0_min)/deltaLam2)

        # initialize arrays
        minimodes = []
        minimodes2 = []
        pIndsDet = []
        tochars = []
        intTimes_all = []
        FA = (len(det) == len(pInds) + 1)
        
        # initialize outputs, and check if there's anything (planet or FA) to characterize
        characterizeds = np.zeros((det.size, num_SNR_bins1 + num_SNR_bins2), dtype=int)
        fZ = 0./u.arcsec**2 * np.ones(nmodes)
        systemParams = SU.dump_system_params(sInd) # write current system params by default
        SNR = np.zeros((len(det), num_SNR_bins1 + num_SNR_bins2))
        intTime = None
        if len(det) == 0: # nothing to characterize
            return characterizeds, fZ, systemParams, SNR, intTime
        
        # look for last detected planets that have not been fully characterized
        for m_i, mode in enumerate(modes):

            if FA is True:
                pIndsDet.append(np.append(pInds, -1)[det])
            else:
                pIndsDet.append(pInds[det])

            if (FA == False): # only true planets, no FA
                tochar = (self.fullSpectra[m_i][pIndsDet[m_i]] == 0)
            else: # mix of planets and a FA
                truePlans = pIndsDet[m_i][:-1]
                tochar = np.append((self.fullSpectra[m_i][truePlans] == 0), True)
        
            # 1/ find spacecraft orbital START position including overhead time,
            # and check keepout angle
            if np.any(tochar):
                # start times
                startTime = TK.currentTimeAbs + mode['syst']['ohTime']
                startTimeNorm = TK.currentTimeNorm + mode['syst']['ohTime']
                # planets to characterize
                tochar[tochar] = Obs.keepout(TL, sInd, startTime, mode)

            # 2/ if any planet to characterize, find the characterization times
            # at the detected fEZ, dMag, and WA
            if np.any(tochar):
                fZ[m_i] = ZL.fZ(Obs, TL, sInd, startTime, mode)
                fEZ = self.lastDetected[sInd,1][det][tochar]/u.arcsec**2
                dMag = self.lastDetected[sInd,2][det][tochar]
                WA = self.lastDetected[sInd,3][det][tochar]*u.arcsec
                intTimes = np.zeros(len(tochar))*u.day
                intTimes[tochar] = OS.calc_intTime(TL, sInd, fZ[m_i], fEZ, dMag, WA, mode)
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

                tochars.append(tochar)
                intTimes_all.append(intTimes)
            else:
                tochar[tochar] = False
                tochars.append(tochar)
        
        # 4/ if yes, perform the characterization for the maximum char time
        if np.any(tochars):

            pIndsChar = []
            TK.allocate_time(modes[0]['syst']['ohTime'])
            for m_i, mode in enumerate(modes):
                if len(pIndsDet[m_i]) > 0:
                    if intTime is None or np.max(intTimes_all[0][tochars[0]]) > intTime:
                        intTime = np.max(intTimes_all[0][tochars[0]])
                    pIndsChar.append(pIndsDet[m_i][tochars[m_i]])
                    log_char = '   - Charact. planet inds %s (%s/%s detected)'%(pIndsChar[m_i], 
                            len(pIndsChar[m_i]), len(pIndsDet[m_i]))
                    self.logger.info(log_char)
                    self.vprint(log_char)
                else:
                    pIndsChar.append([])
            
            # SNR CALCULATION:
            # first, calculate SNR for observable planets (without false alarm)
            if len(pIndsChar[0]) > 0:
                planinds = pIndsChar[0][:-1] if pIndsChar[0][-1] == -1 else pIndsChar[0]
            else: 
                planinds = []
            if len(pIndsChar[1]) > 0:
                planinds2 = pIndsChar[1][:-1] if pIndsChar[1][-1] == -1 else pIndsChar[1]
            else:
                planinds2 = []

            SNRplans = np.zeros((len(planinds), num_SNR_bins1))
            SNRplans2 = np.zeros((len(planinds2), num_SNR_bins2))

            # create a new submode for each spectral bin
            for j in range(num_SNR_bins1):
                minimode = copy.deepcopy(modes[0])
                BW = (1 - ((lam1 - deltaLam/2)/lam1)) * 2
                minimode['lam'] = lam1
                minimode['BW'] = BW
                minimode['deltaLam'] = deltaLam
                minimodes.append(minimode)
                lam1 = copy.deepcopy(lam1) + deltaLam

            for j in range(num_SNR_bins2):
                minimode2 = copy.deepcopy(modes[1])
                BW = (1 - ((lam2 - deltaLam2/2)/lam2)) * 2
                minimode2['lam'] = lam2
                minimode2['BW'] = BW
                minimode2['deltaLam'] = deltaLam2
                minimodes2.append(minimode2)
                lam2 = copy.deepcopy(lam2) + deltaLam2

            if len(planinds) > 0 or len(planinds2) > 0:
                # initialize arrays for SNR integration
                fZs = np.zeros((self.ntFlux, num_SNR_bins1 + num_SNR_bins2))/u.arcsec**2
                systemParamss = np.empty(self.ntFlux, dtype='object')
                Ss = np.zeros((self.ntFlux, len(planinds), num_SNR_bins1))
                Ns = np.zeros((self.ntFlux, len(planinds), num_SNR_bins1))

                Ss2 = np.zeros((self.ntFlux, len(planinds2), num_SNR_bins2))
                Ns2 = np.zeros((self.ntFlux, len(planinds2), num_SNR_bins2))
                # integrate the signal (planet flux) and noise
                dt = intTime/self.ntFlux
                for i in range(self.ntFlux):
                    TK.allocate_time(dt/2.)
                    for j, minimode in enumerate(minimodes):
                        # allocate first half of dt
                        # calculate current zodiacal light brightness
                        fZs[i,j] = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs, minimode)[0]
                        # save planet parameters
                        systemParamss[i] = SU.dump_system_params(sInd)
                        # calculate signal and noise (electron count rates)
                        Ss[i,:,j], Ns[i,:,j] = self.calc_signal_noise(sInd, planinds, dt, minimode, 
                                               fZ=fZs[i,j])

                    for j2, minimode in enumerate(minimodes2):
                        # allocate first half of dt
                        # calculate current zodiacal light brightness
                        fZs[i,j + j2] = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs, minimode)[0]
                        # calculate signal and noise (electron count rates)
                        Ss2[i,:,j2], Ns2[i,:,j2] = self.calc_signal_noise(sInd, planinds2, dt, minimode, 
                                                   fZ=fZs[i, j + j2])

                    self.propagTimes[sInd] = TK.currentTimeNorm
                    # propagate the system to match up with current time
                    SU.propag_system(sInd, TK.currentTimeNorm - self.propagTimes[sInd])
                    # allocate second half of dt
                    TK.allocate_time(dt/2.)
                
                # average output parameters
                fZ = np.mean(fZs)
                systemParams = {key: sum([systemParamss[x][key]
                        for x in range(self.ntFlux)])/float(self.ntFlux)
                        for key in sorted(systemParamss[0])}
                # calculate planets SNR
                S = Ss.sum(0)
                N = Ns.sum(0)
                S2 = Ss2.sum(0)
                N2 = Ns2.sum(0)
                SNRplans[N > 0] = S[N > 0]/N[N > 0]
                SNRplans2[N2 > 0] = S2[N2 > 0]/N2[N2 > 0]
                # allocate extra time for timeMultiplier
                extraTime = intTime*(mode['timeMultiplier'] - 1)
                TK.allocate_time(extraTime)
            
            # if only a FA, just save zodiacal brightness in the middle of the integration
            else:
                totTime = intTime*(mode['timeMultiplier'])
                TK.allocate_time(totTime/2.)
                for m_i, mode in enumerate(modes):
                    fZ[m_i] = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs, mode)[0]
                TK.allocate_time(totTime/2.)
            
            # calculate the false alarm SNR (if any)
            for m_i, mode in enumerate(modes):
                if len(pIndsChar[m_i]) > 0:
                    SNRfa = []
                    if pIndsChar[m_i][-1] == -1:
                        fEZ = self.lastDetected[sInd,1][-1]/u.arcsec**2
                        dMag = self.lastDetected[sInd,2][-1]
                        WA = self.lastDetected[sInd,3][-1]*u.arcsec
                        C_p, C_b, C_sp = OS.Cp_Cb_Csp(TL, sInd, fZ[m_i], fEZ, dMag, WA, mode)
                        S = (C_p*intTime).decompose().value
                        N = np.sqrt((C_b*intTime + (C_sp*intTime)**2).decompose().value)
                        SNRfa.append([S/N if N > 0 else 0.])
                
                    # save all SNRs (planets and FA) to one array
                    SNRinds = np.where(det)[0][tochars[m_i]]
                    if m_i == 0:
                        SNR[SNRinds, :num_SNR_bins1] =  np.append(SNRplans[:], SNRfa)
                    else:
                        SNR[SNRinds, :num_SNR_bins2] =  np.append(SNRplans2[:], SNRfa)
                
                    # now, store characterization status: 1 for full spectrum, 
                    # -1 for partial spectrum, 0 for not characterized
                    char = (SNR[:,m_i] >= mode['SNR'])
                    # initialize with full spectra
                    characterized = char.astype(int)
                    WAchar = self.lastDetected[sInd,3][char]*u.arcsec
                    # find the current WAs of characterized planets
                    WAs = systemParams['WA']
                    if FA:
                        WAs = np.append(WAs, self.lastDetected[sInd,3][-1]*u.arcsec)
                    # check for partial spectra
                    IWA_max = mode['IWA']*(1 + mode['BW']/2.)
                    OWA_min = mode['OWA']*(1 - mode['BW']/2.)
                    char[char] = (WAchar < IWA_max) | (WAchar > OWA_min)
                    characterized[char] = -1
                    # encode results in spectra lists (only for planets, not FA)
                    charplans = characterized[:-1] if FA else characterized
                    self.fullSpectra[m_i][pInds[charplans == 1]] += 1
                    self.partialSpectra[m_i][pInds[charplans == -1]] += 1
                    characterizeds[:,m_i] = characterized.astype(int)

        return characterized.astype(int), fZ, systemParams, SNR, intTime
