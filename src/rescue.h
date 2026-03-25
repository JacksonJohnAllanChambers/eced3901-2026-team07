/* RESCUE MODULE - TEAM 7*/

#ifndef RESCUE_H
#define RESCUE_H

// includes

#include <avr/io.h>
#include <avr/interrupt.h>

// rescue parameters

#define RESCUE_POS_START 3000
#define RESCUE_POS_TRAP 1000

#define RESCUE_SERVO_PIN PINB2
#define RESCUE_VIRTUAL_PIN PINC1

// define enum

typedef enum {
    RESCUE_START,
    RESCUE_MOVING,
    RESCUE_HOLD,
    RESCUE_RETURNING
} RescueState;

// define rescue functions

void rescue_init(void);
void rescue_boat(char aligned);
void virtual_pwm_tick(void);

#endif