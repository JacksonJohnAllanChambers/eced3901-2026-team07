/* LIGHTING MODULE IMPLEMENTATION*/

#include <avr/io.h>
#include <avr/interrupt.h>
#include <stdlib.h>
#include "fsk.h"

void lighting_init(void){

    /* Scheduling Timer Setup */

    // Enable CTC mode
    TCCR0A = (1<<WGM01);
    // Set COMPA value for 20us timer
    OCR0A = 39; 
    // Initialize timer register
    TCNT0 = 0;
    // Enable COMPA interrupts
    TIMSK0 |= (1<<OCIE0A);
    // Set prescaler to 8
    TCCR0B |= (1<<CS01);

    /* FSK Setup */
    
    // Enable OC2B as output
    DDRD |= (1<<FSK_OUTPUT_PIN);
    // Enable pull-up on electromagnet control pin
    PORTD |= (1<<CONTROL_PIN);
    // Enable non-inverting Fast PWM, OCR2A TOP, prescaler 64
    TCCR2A |= (1<<WGM20) | (1<<WGM21) | (1<<COM2B1);
    TCCR2B |= (1<<WGM22) | (1<<CS11);

    /* Ultrasonic Setup */

    // Enable pinout
    DDRB |= (1<<TRIG_PIN) | (1<<LED_RED_PIN) | (1<<LED_GREEN_PIN) | (1<<LED_YELLOW_PIN);

}

void fsk_handler(void){

    // Handle bit timing

    if(count >= 454){ // ~9ms (bit period) has passed
        next_bit = 1;
        count = 0;
    }



}
