/* RESCUE MODULE - TEAM 7*/

#ifndef CARGO_H
#define CARGO_H

// includes

#include <stdint.h>

// cargo parameters

#define POS_MAX 5000
#define POS_START 4000
#define POS_RECOV 3500
#define CARGO_SERVO_PIN PINB1
#define MAGNET_ACTIVATION_PIN PINC3

// declare cargo functions

void cargo_init(void);
void pickup_cargo(void);

#endif