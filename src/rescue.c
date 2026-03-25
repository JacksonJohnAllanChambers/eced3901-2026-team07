/* RESCUE MODULE IMPLEMENTATION */

#include "rescue.h"
#include "schedule.h"

// defined depth variable
static uint32_t state_timer = 0;
static RescueState state = RESCUE_START;
static uint16_t pwm_count = 0;
static volatile uint16_t virtual_servo_pos = 50;

void rescue_init(void){
    // Set OC1B as output (PB2)
    DDRB |= (1<<RESCUE_SERVO_PIN);
    DDRC |= (1<<RESCUE_VIRTUAL_PIN);

    // Enable non-inverting PWM on OC1B
    // OR into TCCR1A to preserve existing OC1A and WGM bits
    TCCR1A |= (1<<COM1B1);

    // Reset virtual PWM state
    pwm_count = 0;
    PORTC |= (1<<RESCUE_VIRTUAL_PIN);  // start pin high

    // Set start position
    OCR1B = RESCUE_POS_START;
    virtual_servo_pos = 50;

    state = RESCUE_START;
    state_timer = 0;
}

void rescue_boat(char activate){

    switch(state){

        case RESCUE_START:
            // move to rescue position if commanded
            if(activate == 1){
                state_timer = ccount;
                state = RESCUE_MOVING;
            }
            break;

        case RESCUE_MOVING:
            // small delay to let servo settle
            if(ccount - state_timer >= 2500){
                OCR1B = RESCUE_POS_TRAP;
                virtual_servo_pos = 100;
                state_timer = ccount;
                state = RESCUE_HOLD;
            }
            break;

        case RESCUE_HOLD:
            // hold trap position until commanded to return
            if(activate == 0){
                state_timer = ccount;
                state = RESCUE_RETURNING;
            }
            break;

        case RESCUE_RETURNING:
            // small delay to let servo settle
            if(ccount - state_timer >= 2500){
                OCR1B = RESCUE_POS_START;
                virtual_servo_pos = 50;
                state_timer = ccount;
                state = RESCUE_START;
            }
            break;

    }  

}

void virtual_pwm_tick(void){
    pwm_count++;

    if(pwm_count >= virtual_servo_pos){
        PORTC &= ~(1<<RESCUE_VIRTUAL_PIN); // pin low
    }

    if(pwm_count >= 1000){
        pwm_count = 0;
        PORTC |= (1<<RESCUE_VIRTUAL_PIN);  // pin high at start of new period
    }
}