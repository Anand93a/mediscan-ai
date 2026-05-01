const jwt = require("jsonwebtoken");
const User = require("../models/User");

/**
 * Middleware: Verify JWT and attach user to request.
 * If token is missing or invalid, req.user will be null (allows optional auth).
 */
const authenticate = async (req, res, next) => {
  try {
    const header = req.headers.authorization;
    if (!header || !header.startsWith("Bearer ")) {
      req.user = null;
      return next();
    }

    const token = header.split(" ")[1];
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    const user = await User.findById(decoded.id).select("-password");

    if (!user) {
      req.user = null;
      return next();
    }

    req.user = user;
    next();
  } catch (error) {
    req.user = null;
    next();
  }
};

/**
 * Middleware: Require authenticated user.
 */
const requireAuth = (req, res, next) => {
  if (!req.user) {
    return res.status(401).json({ error: "Authentication required" });
  }
  next();
};

/**
 * Middleware: Require admin role.
 */
const requireAdmin = (req, res, next) => {
  if (!req.user) {
    return res.status(401).json({ error: "Authentication required" });
  }
  if (req.user.role !== "admin") {
    return res.status(403).json({ error: "Admin access required" });
  }
  next();
};

module.exports = { authenticate, requireAuth, requireAdmin };
