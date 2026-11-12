/* Pre-change ObjectMove.s phase scanning and substep wrappers from
 * bd5f6ed:src/movement.c. Keep these separate from the optimized primary
 * loops so full context/room/floorline comparisons cannot use that shortcut
 * on both sides. Shared collision helpers retain their original behavior;
 * touch arithmetic also has an independent volatile wide-division oracle. */

static void reference_move_object(MoveContext* ctx, LevelState* level)
{
    if (!level || !level->data || !level->floor_lines) {
        ctx->hitwall = 0;
        ctx->wall_hit_y = ctx->newy;
        return;
    }

    ctx->hitwall = 0;
    ctx->wall_hit_y = ctx->newy;
    ctx->wall_xsize = 0;
    ctx->wall_zsize = 0;
    ctx->wall_length = 0;

    {
        int32_t xdiff = ctx->newx - ctx->oldx;
        int32_t zdiff = ctx->newz - ctx->oldz;
        if (xdiff == 0 && zdiff == 0) return;

        const uint8_t* zone_data = ctx->objroom;

        int total_iterations = 0;
        const int max_total = 200;

        if (zone_data) {
        gobackanddoitallagain:
            if (total_iterations >= max_total) return;

            if (total_iterations >= max_total) goto phase3;

            {
                int16_t list_off = read_be16(zone_data + ZONE_EXIT_LIST);
                const uint8_t* list_ptr = zone_data + list_off;

                for (int i = 0; i < 128; i++, total_iterations++) {
                    if (total_iterations >= max_total) goto phase3;

                    int16_t entry = read_be16(list_ptr + i * 2);
                    if (entry < 0) break; /* Amiga checkwalls: stop at first negative */
                    if (entry >= level->num_floor_lines) continue;

                    {
                        const uint8_t* fline = level->floor_lines + entry * FLINE_SIZE;
                        int result = check_wall_line(ctx, level, fline, zone_data, &xdiff, &zdiff, MOVE_PASS_WALLS);

                        /* Amiga MoveObject continues scanning after a slide-adjusted hit
                         * (hitthewall -> checkwalls/checkotherwalls), it does not restart
                         * the pass from the beginning. */
                        if (result == 2) continue;
                        if (result == 3) goto phase3;
                    }
                }
            }

            if (ctx->extlen != 0) {
                if (total_iterations >= max_total) goto phase3;

                {
                    int16_t list_off = read_be16(zone_data + ZONE_EXIT_LIST);
                    const uint8_t* list_ptr = zone_data + list_off;

                    for (int i = 0; i < 128; i++, total_iterations++) {
                        if (total_iterations >= max_total) goto phase3;

                        int16_t entry = read_be16(list_ptr + i * 2);
                        if (entry < 0) {
                            if (entry == -2) break; /* Amiga checkotherwalls terminator */
                            continue;               /* -1 separator */
                        }
                        if (entry >= level->num_floor_lines) continue;

                        {
                            const uint8_t* fline = level->floor_lines + entry * FLINE_SIZE;
                            int result = check_wall_line(ctx, level, fline, zone_data, &xdiff, &zdiff, MOVE_PASS_OTHER);

                            if (result == 2) continue;
                            if (result == 3) goto phase3;
                        }
                    }
                }
            }

        phase3:
            {
                const uint8_t* prev_zone = zone_data;
                find_room(ctx, level, &zone_data);

                if (zone_data != prev_zone) {
                    goto gobackanddoitallagain;
                }
            }
        }
        else {
            int32_t num_lines = (int32_t)level->num_floor_lines;
            if (num_lines <= 0) return;

            if (total_iterations >= max_total) return;

            for (int32_t i = 0; i < num_lines; i++, total_iterations++) {
                if (total_iterations >= max_total) return;

                {
                    const uint8_t* fline = level->floor_lines + i * FLINE_SIZE;
                    int result = check_wall_line(ctx, level, fline, zone_data, &xdiff, &zdiff, MOVE_PASS_BRUTE);

                    if (result == 2) continue;
                    if (result == 3) return;
                }
            }
        }
    }
}

static void reference_move_object_substepped(MoveContext* ctx, LevelState* level)
{
    int32_t start_x = ctx->oldx;
    int32_t start_z = ctx->oldz;
    int32_t target_x = ctx->newx;
    int32_t target_z = ctx->newz;
    int32_t dx = target_x - start_x;
    int32_t dz = target_z - start_z;
    int32_t adx = (dx < 0) ? -dx : dx;
    int32_t adz = (dz < 0) ? -dz : dz;
    int32_t maxd = (adx > adz) ? adx : adz;
    int steps = (int)(maxd / 4) + 1;
    if (steps > 32) steps = 32;

    int32_t cur_x = start_x;
    int32_t cur_z = start_z;
    uint8_t* cur_room = ctx->objroom;
    int8_t cur_top = ctx->stood_in_top;
    int8_t any_hit = 0;
    int32_t hit_y = ctx->newy;
    int16_t hit_wx = 0;
    int16_t hit_wz = 0;
    int16_t hit_wlen = 0;

    for (int i = 1; i <= steps; i++) {
        MoveContext step = *ctx;
        step.objroom = cur_room;
        step.stood_in_top = cur_top;
        step.oldx = cur_x;
        step.oldz = cur_z;
        step.newx = start_x + (dx * i) / steps;
        step.newz = start_z + (dz * i) / steps;
        step.xdiff = step.newx - step.oldx;
        step.zdiff = step.newz - step.oldz;

        reference_move_object(&step, level);

        if (step.hitwall) {
            any_hit = 1;
            hit_y = step.wall_hit_y;
            hit_wx = step.wall_xsize;
            hit_wz = step.wall_zsize;
            hit_wlen = step.wall_length;
            if (step.exitfirst) {
                cur_x = step.newx;
                cur_z = step.newz;
                cur_room = step.objroom;
                cur_top = step.stood_in_top;
                break;
            }
        }
        cur_x = step.newx;
        cur_z = step.newz;
        cur_room = step.objroom;
        cur_top = step.stood_in_top;
    }

    ctx->newx = cur_x;
    ctx->newz = cur_z;
    ctx->objroom = cur_room;
    ctx->stood_in_top = cur_top;
    ctx->hitwall = any_hit;
    ctx->wall_hit_y = hit_y;
    ctx->wall_xsize = hit_wx;
    ctx->wall_zsize = hit_wz;
    ctx->wall_length = hit_wlen;
}
