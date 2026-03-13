#include "schedule.h"
#include "cargo.h"
#include "lighting.h"
#include <avr/interrupt.h>


ISR(TIMER0_COMPA_vect){
    scheduler_tick();
    ultrasonic_tick();
    fsk_handler();
}

int main(void){
    // initialize variables
    char cargo_pickup = 0, check = 0;
    // pull up
    PORTC |= (1<<PINC2);

    // initialize i/o pins and registers
    lighting_init(); 
    cargo_init();

    // enable global interrupts
    sei();

    while(1){
        // continuously poll ultrasonic sensor
        ultrasonic_update();
        LED_update();
        cargo_update(cargo_pickup);
        cargo_pickup = 0;
        if(!(PINC & (1 << PINC2))){
            cargo_pickup = 1;
            check = 1;
        }
    }
}