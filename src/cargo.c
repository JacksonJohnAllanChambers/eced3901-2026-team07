/* CARGO MODULE IMPLEMENTATION */

#include "cargo.h"
#include "lighting.h"

// defined depth variable
static uint16_t TARGET_DEPTH = 1000; 

void cargo_init(void){

    // Set OC1A as output
    DDRB |= (1<<CARGO_SERVO_PIN);
    // Set magnet control pin as output
    DDRC |= (1<<MAGNET_ACTIVATION_PIN);

    // Set non-inverting 50Hz Fast PWM, ICR1 TOP, 8 prescaler
    TCCR1A = (1<<COM1A1) | (1<<WGM11);
    TCCR1B = (1<<WGM13) | (1<<WGM12) | (1<<CS11);
    ICR1 = 39999;
    // Set start position
    OCR1A = POS_START;
}

void pickup_cargo(void){

    // Turn on electromagnets
    PORTC |= (1<<MAGNET_ACTIVATION_PIN);

}
