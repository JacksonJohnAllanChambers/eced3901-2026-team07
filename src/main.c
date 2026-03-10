#include "lighting.h"

ISR(TIMER0_COMPA_vect){
    ultrasonic_tick();
    fsk_handler();
}

int main(void){

    // initialize i/o pins and registers
    lighting_init();

    while(1){
        // continuously poll ultrasonic sensor
        ultrasonic_update();
        LED_update();
    }
}