
    def normalized_cross_correlation(self, raw_tile, template):
        template = template.copy()
        _, raw_tile_contours, _ = cv2.findContours(raw_tile, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _, template_contours, _ = cv2.findContours(template, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        tile_areas = [-cv2.contourArea(c) for c in raw_tile_contours]
        template_areas = [-cv2.contourArea(c) for c in template_contours]

        print('tile areas:', tile_areas)
        print('templat eareas:', template_areas)
        for i, j in zip(np.argsort(tile_areas), np.argsort(template_areas)):
            tile_c = raw_tile_contours[i]
            template_c = template_contours[j]

            epsilon = 3
            tile_c = cv2.approxPolyDP(tile_c,epsilon,True)
            template_c = cv2.approxPolyDP(template_c, epsilon, True)

            similarity = cv2.matchShapes(tile_c, template_c, cv2.CONTOURS_MATCH_I1, 0.0)
            print('raw tile contours len', len(raw_tile_contours))
            print('template_contours len', len(template_contours))
            print('similarity:', similarity)
            print('raw tile contour area:', tile_areas[i])
            print('template tile contour area:', template_areas[j])
            raw_tile_copy = np.copy(raw_tile)
            raw_tile_copy = cv2.cvtColor(raw_tile_copy, cv2.COLOR_GRAY2BGR)
            template_copy = np.copy(template)
            template_copy = cv2.cvtColor(template_copy, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(raw_tile_copy, [tile_c], -1, (0, 0, 255))
            cv2.drawContours(template_copy, [template_c], -1, (0, 0, 255))
            show_images([raw_tile_copy, template_copy])

        normalized_tile = self.normalized_img(raw_tile)
        normalized_template = self.normalized_img(template)

        max_score = -1.0
        diff_y = normalized_tile.shape[0] - normalized_template.shape[0]
        diff_x = normalized_tile.shape[1] - normalized_template.shape[1]
        H, W = normalized_template.shape

        for y, x in np.ndindex((diff_y + 1, diff_x + 1)):
            window = normalized_tile[y:y+H, x:x+W]
            #print('sum:', (window * normalized_template).sum())
            max_score = max(max_score, (window * normalized_template).sum())

        # res = cv2.matchTemplate(raw_tile, template, cv2.TM_CCORR_NORMED)
        # min_val, max_score, min_loc, max_loc = cv2.minMaxLoc(res)
        #print('max score:', max_score)

        return max_score