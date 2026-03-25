/* RESCUE MODULE - TEAM 7*/

#ifndef CARGO_H
#define CARGO_H

// includes

#include <stdint.h>

// cargo parameters

#define POS_MAX 5000
#define POS_START 4000
#define POS_RECOV 6000
#define TARGET_DEPTH 1000
#define CARGO_SERVO_PIN PINB1
#define MAGNET_ACTIVATION_PIN PINC3

// state machine declaration

typedef enum{
    
    CARGO_IDLE,
    ACTIVATE_MAGNET,
    SERVO_LOWER,
    SERVO_DELAY,
    SERVO_RAISE,
    CARGO_RELEASE

} CargoState;

// declare cargo functions

void cargo_init(void);
void cargo_update(char aligned);

#endif