

flagdata(vis='3c391_ctm_mosaic_10s_spw0.ms', flagbackup=True, mode='manual', scan='1')
flagdata(vis='3c391_ctm_mosaic_10s_spw0.ms', flagbackup=True, mode='manual', antenna='ea13,ea15')
flagdata(vis='3c391_ctm_mosaic_10s_spw0.ms', mode='quack', quackinterval=10.0, quackmode='beg')
gencal(vis='3c391_ctm_mosaic_10s_spw0.ms',caltable='3c391_ctm_mosaic_10s_spw0.antpos',caltype='antpos')
setjy(vis='3c391_ctm_mosaic_10s_spw0.ms',field='J1331+3030',standard='Perley-Butler 2017',
      model='3C286_C.im',usescratch=True,scalebychan=True,spw='')
gaincal(vis='3c391_ctm_mosaic_10s_spw0.ms', caltable='3c391_ctm_mosaic_10s_spw0.G0all', 
        field='0,1,9', refant='ea21', spw='0:27~36',
        gaintype='G',calmode='p', solint='int', 
        minsnr=5, gaintable=['3c391_ctm_mosaic_10s_spw0.antpos'])
flagdata(vis='3c391_ctm_mosaic_10s_spw0.ms',
         flagbackup=True, mode='manual', antenna='ea05')
gaincal(vis='3c391_ctm_mosaic_10s_spw0.ms', caltable='3c391_ctm_mosaic_10s_spw0.G0', 
        field='J1331+3030', refant='ea21', spw='0:27~36', calmode='p', solint='int', 
        minsnr=5, gaintable=['3c391_ctm_mosaic_10s_spw0.antpos'])
gaincal(vis='3c391_ctm_mosaic_10s_spw0.ms',caltable='3c391_ctm_mosaic_10s_spw0.K0', 
        field='J1331+3030',refant='ea21',spw='0:5~58',gaintype='K', 
        solint='inf',combine='scan',minsnr=5,
        gaintable=['3c391_ctm_mosaic_10s_spw0.antpos',
                   '3c391_ctm_mosaic_10s_spw0.G0'])
bandpass(vis='3c391_ctm_mosaic_10s_spw0.ms',caltable='3c391_ctm_mosaic_10s_spw0.B0',
         field='J1331+3030',spw='',refant='ea21',combine='scan', 
         solint='inf',bandtype='B',
         gaintable=['3c391_ctm_mosaic_10s_spw0.antpos',
                    '3c391_ctm_mosaic_10s_spw0.G0',
                    '3c391_ctm_mosaic_10s_spw0.K0'])
gaincal(vis='3c391_ctm_mosaic_10s_spw0.ms',caltable='3c391_ctm_mosaic_10s_spw0.G1',
        field='J1331+3030',spw='0:5~58',
        solint='inf',refant='ea21',gaintype='G',calmode='ap',solnorm=False,
        gaintable=['3c391_ctm_mosaic_10s_spw0.antpos',
                   '3c391_ctm_mosaic_10s_spw0.K0',
                   '3c391_ctm_mosaic_10s_spw0.B0'],
        interp=['','','nearest'])
gaincal(vis='3c391_ctm_mosaic_10s_spw0.ms',caltable='3c391_ctm_mosaic_10s_spw0.G1',
        field='J1822-0938',
        spw='0:5~58',solint='inf',refant='ea21',gaintype='G',calmode='ap',
        gaintable=['3c391_ctm_mosaic_10s_spw0.antpos',
                   '3c391_ctm_mosaic_10s_spw0.K0',
                   '3c391_ctm_mosaic_10s_spw0.B0'],
        append=True)
myscale = fluxscale(vis='3c391_ctm_mosaic_10s_spw0.ms',
                    caltable='3c391_ctm_mosaic_10s_spw0.G1', 
                    fluxtable='3c391_ctm_mosaic_10s_spw0.fluxscale1', 
                    reference='J1331+3030',
                    transfer=['J1822-0938'],
                    incremental=False)
applycal(vis='3c391_ctm_mosaic_10s_spw0.ms',
         field='J1331+3030',
         gaintable=['3c391_ctm_mosaic_10s_spw0.antpos', 
                    '3c391_ctm_mosaic_10s_spw0.fluxscale1',
                    '3c391_ctm_mosaic_10s_spw0.K0',
                    '3c391_ctm_mosaic_10s_spw0.B0'],
         gainfield=['','J1331+3030','',''], 
         interp=['','nearest','',''],
         calwt=False)

applycal(vis='3c391_ctm_mosaic_10s_spw0.ms',
         field='J1822-0938',
         gaintable=['3c391_ctm_mosaic_10s_spw0.antpos', 
                    '3c391_ctm_mosaic_10s_spw0.fluxscale1',
                    '3c391_ctm_mosaic_10s_spw0.K0',
                    '3c391_ctm_mosaic_10s_spw0.B0'],
         gainfield=['','J1822-0938','',''], 
         interp=['','nearest','',''],
         calwt=False)
applycal(vis='3c391_ctm_mosaic_10s_spw0.ms',
         field='2~8',
         gaintable=['3c391_ctm_mosaic_10s_spw0.antpos', 
                    '3c391_ctm_mosaic_10s_spw0.fluxscale1',
                    '3c391_ctm_mosaic_10s_spw0.K0',
                    '3c391_ctm_mosaic_10s_spw0.B0'],
         gainfield=['','J1822-0938','',''], 
         interp=['','linear','',''],
         calwt=False)