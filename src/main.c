#include "schedule.h"
#include "lighting.h"
#include <avr/interrupt.h>


ISR(TIMER0_COMPA_vect){
    scheduler_tick();
    ultrasonic_tick();
    fsk_handler();
}

int main(void){

    // initialize i/o pins and registers
    lighting_init();

    // enable global interrupts
    sei();

    while(1){
        // continuously poll ultrasonic sensor
        ultrasonic_update();
        LED_update();
    }
}