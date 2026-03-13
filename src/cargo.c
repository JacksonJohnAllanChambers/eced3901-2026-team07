/* CARGO MODULE IMPLEMENTATION */

#include "schedule.h"
#include "cargo.h"
#include "lighting.h"

// defined depth variable
static uint16_t current_pos = 0; 
static uint32_t state_timer = 0;

// initialize state machine
static CargoState state = CARGO_IDLE;

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

void cargo_update(char aligned){

    // implement state machine
    switch(state){

        case CARGO_IDLE:
            // change state if cargo is aligned
            if(aligned == 1){
                // Turn on electromagnets
                PORTC |= (1<<MAGNET_ACTIVATION_PIN);
                state = ACTIVATE_MAGNET;
                state_timer = ccount;
            }
            // release cargo immediately
            if(aligned == 2){
                state = CARGO_RELEASE;
                state_timer = ccount;
            } 
            break;

        case ACTIVATE_MAGNET:
            // 50ms delay
            if(ccount - state_timer >= 2500){
                state_timer = ccount;
                current_pos = POS_START;
                state = SERVO_LOWER;
            }
            break;
        
        case SERVO_LOWER:
            // 5ms smoothing delay
            if(ccount - state_timer >= 250){
                // Have magnets reached target depth?
                if(current_pos > TARGET_DEPTH){
                    current_pos -= 20;
                    OCR1A = current_pos;
                    state_timer = ccount;
                }
                else{
                    state_timer = ccount;
                    state = SERVO_DELAY;
                }
            }
            break;

        case SERVO_DELAY:
            // 1 second delay
            if(ccount - state_timer >= 50000){
                state_timer = ccount;
                state = SERVO_RAISE;
            }
            break;

        case SERVO_RAISE:
            // 5ms smoothing delay
            if(ccount - state_timer >= 250){
                // Has cargo been lifted from the ground?
                if(current_pos < POS_RECOV){
                    current_pos += 20;
                    OCR1A = current_pos;
                    state_timer = ccount;
                }
                else{
                    state_timer = ccount;
                    state = CARGO_IDLE;
                }
            }
            break;

        case CARGO_RELEASE:
            // Turn off electromagnets
            PORTC &= ~(1<<MAGNET_ACTIVATION_PIN);
            state = CARGO_IDLE;
            break;

    }

}
