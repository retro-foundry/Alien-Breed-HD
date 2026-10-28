/* Focused regression check; run with tools/check_feedback.py. */
#include "../src/audio.c"
#include <assert.h>
#undef printf
static double tone(int rate, int hz, int submerged) {
    g_spec.freq=rate; audio_filter_init(); g_underwater=submerged;
    double energy=0; int n=rate/2;
    for(int i=0;i<n;i++) {
        Sint16 sample=(Sint16)(12000*sin(6.283185307179586*hz*i/rate));
        Sint16 dry=sample;
        audio_filter_output((Uint8 *)&sample,sizeof(sample));
        if(!submerged) assert(sample==dry);
        if(i>rate/10) energy+=(double)sample*sample;
    }
    return sqrt(energy/(n-rate/10-1));
}
int main(void) {
    /* SoundPlayer.s MakeSomeNoise, Noisevol=200 (Anims.s ItsAGasPipe). */
    assert(audio_distance_volume(200,0,0)==255);
    assert(audio_distance_volume(200,800,0)==253);
    assert(audio_distance_volume(200,1600,0)==127);
    assert(audio_distance_volume(200,3200,0)==63);
    assert(audio_distance_volume(200,0,-3200)==63);
    assert(audio_distance_volume(200,1000000,0)==0);
    assert(audio_distance_volume(0,0,0)==0);
    int changes=0;
    for(int distance=800;distance<16000;distance++) {
        int a=audio_distance_volume(200,distance,0);
        int b=audio_distance_volume(200,distance+1,0);
        assert(a>=b && a-b<=1);
        if(distance>=1600 && distance<3200 && a!=b) changes++;
    }
    assert(changes>=60); /* Former Paula rounding supplied only 16 steps here. */
    static Uint8 sample[100];
    g_samples[22].data=sample; g_samples[22].length=sizeof(sample);
    for(int nearer_first=0;nearer_first<2;nearer_first++) {
        memset(g_channels,0,sizeof(g_channels));
        audio_begin_frame();
        audio_queue_sample(22,nearer_first?251:59);
        g_channels[0].position=12;
        audio_queue_sample(22,nearer_first?59:251);
        assert(g_channels[0].volume==251 && g_channels[0].position==12);
        assert(channel_is_free(&g_channels[1]));
        audio_begin_frame();
        audio_queue_sample(22,59);
        assert(g_channels[1].volume==59 && g_channels[1].position==0);
    }
    puts("PASS: smooth fire-spout distance curve; simultaneous emitters keep loudest burst without restarting");
    int rates[]={8007,48000};
    for(int r=0;r<2;r++) {
        int rate=rates[r];
        double low=tone(rate,300,1)/tone(rate,300,0);
        double high=tone(rate,rate==8007?3800:10000,1)/tone(rate,rate==8007?3800:10000,0);
        double cutoff=tone(rate,2200,1)/tone(rate,2200,0);
        double mid=tone(rate,2500,1)/tone(rate,2500,0);
        assert(low>0.98 && low<1.02);
        assert(high<0.15);
        assert(cutoff>0.70 && cutoff<0.72);
        assert(mid<0.65); /* More muffling within the original samples' band. */
        printf("PASS rate=%d: low gain %.3f high gain %.3f; dry bypass exact\n",rate,low,high);
        g_spec.freq=rate; audio_filter_init(); g_underwater=1;
        Sint16 samples[2048]={0};
        audio_filter_output((Uint8*)samples,sizeof(samples));
        assert(g_filter_wet==1.0f);
        g_underwater=0;
        for(int j=0;j<2;j++) audio_filter_output((Uint8*)samples,sizeof(samples));
        assert(g_filter_wet==0.0f);
        for(int j=0;j<2048;j++) assert(samples[j]==0);
    }
}
